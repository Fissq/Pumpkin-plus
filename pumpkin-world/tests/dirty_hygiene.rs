//! Dirty-hygiene: reading already-generated territory must not rewrite region files.
//!
//! Scenario (flat world for cheap generation, proto stages only):
//! 1. Generate an area up to the Features stage, shut the level down
//!    (everything flushed to disk).
//! 2. Warm-up revisit: reopen the level over the same directory, load the same
//!    area, shut down. This absorbs any chunks the first shutdown raced past,
//!    so the on-disk state is complete.
//! 3. Revisit again without modifications, explicitly exercise the unload
//!    path, shut down — region files must be untouched (len + mtime + bytes).
//!
//! Counter-test: a revisit that *advances* proto stages must write.
//!
//! NOTE: ticket levels <= 43 (Full chunks) are deliberately avoided here —
//! flat worlds currently never finish the Lighting/Spawn/Full stages through
//! the scheduler (pre-existing hang, reproduced on unpatched master; see
//! docs/baseline/dirty-hygiene-2026-07-16.md). Proto stages are exactly what
//! this PR is about, so the coverage is unaffected.

use pumpkin_config::chunk::ChunkConfig;
use pumpkin_config::lighting::LightingEngineConfig;
use pumpkin_config::world::LevelConfig;
use pumpkin_data::BlockStateId;
use pumpkin_data::dimension::Dimension;
use pumpkin_nbt::compound::NbtCompound;
use pumpkin_nbt::tag::NbtTag;
use pumpkin_util::math::vector2::Vector2;
use pumpkin_world::level::Level;
use pumpkin_world::world::WorldPortalExt;
use pumpkin_world::world_info::data_files::write_world_gen_settings;
use pumpkin_world::world_info::{
    Dimension as WgDimension, Generator, GeneratorSettings, WorldGenSettings,
};
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Duration;

const SEED: i64 = 42;
/// Ticket level that drives the ticket position to the Features stage
/// (see `StagedChunkEnum::level_to_stage`).
const FEATURES_TICKET: i8 = 46;
/// Ticket level that drives the ticket position to the Carvers stage.
const CARVERS_TICKET: i8 = 47;

struct BlockRegistry;
impl WorldPortalExt for BlockRegistry {
    fn can_place_at(
        &self,
        _block: &pumpkin_data::Block,
        _state: &pumpkin_data::BlockState,
        _block_accessor: &dyn pumpkin_world::world::BlockAccessor,
        _block_pos: &pumpkin_util::math::position::BlockPos,
    ) -> bool {
        true
    }

    fn mirror(
        &self,
        block: &pumpkin_data::Block,
        state_id: BlockStateId,
        mirror: pumpkin_data::Mirror,
    ) -> &'static pumpkin_data::BlockState {
        block.mirror(state_id, mirror)
    }

    fn rotate(
        &self,
        block: &pumpkin_data::Block,
        state_id: BlockStateId,
        rotation: pumpkin_data::Rotation,
    ) -> &'static pumpkin_data::BlockState {
        block.rotate(state_id, rotation)
    }

    fn spawn_mobs_for_chunk_generation(
        &self,
        _cache: &mut dyn pumpkin_world::generation::proto_chunk::GenerationCache,
        _biome: &'static pumpkin_data::chunk::Biome,
        _chunk_x: i32,
        _chunk_z: i32,
    ) {
    }
}

/// Writes world_gen_settings.dat selecting the flat generator so chunk
/// generation in the test is nearly free.
fn write_flat_world_settings(root: &Path) {
    let mut layer_bedrock = NbtCompound::new();
    layer_bedrock.put("block", NbtTag::String("minecraft:bedrock".into()));
    layer_bedrock.put("height", NbtTag::Int(1));
    let mut layer_dirt = NbtCompound::new();
    layer_dirt.put("block", NbtTag::String("minecraft:dirt".into()));
    layer_dirt.put("height", NbtTag::Int(2));
    let mut layer_grass = NbtCompound::new();
    layer_grass.put("block", NbtTag::String("minecraft:grass_block".into()));
    layer_grass.put("height", NbtTag::Int(1));

    let mut settings = NbtCompound::new();
    settings.put("biome", NbtTag::String("minecraft:plains".into()));
    settings.put(
        "layers",
        NbtTag::List(vec![
            NbtTag::Compound(layer_bedrock),
            NbtTag::Compound(layer_dirt),
            NbtTag::Compound(layer_grass),
        ]),
    );

    let mut dimensions = std::collections::HashMap::new();
    dimensions.insert(
        "minecraft:overworld".to_string(),
        WgDimension {
            generator: Generator {
                settings: Some(GeneratorSettings::Compound(settings)),
                biome_source: None,
                generator_type: "minecraft:flat".to_string(),
            },
            dimension_type: "minecraft:overworld".to_string(),
        },
    );

    let wgs = WorldGenSettings {
        seed: SEED,
        dimensions,
    };
    write_world_gen_settings(root, &wgs, 0).expect("failed to write world_gen_settings.dat");
}

fn level_config() -> LevelConfig {
    LevelConfig {
        chunk: ChunkConfig::default(),
        lighting: LightingEngineConfig::Default,
        autosave_ticks: 0,
    }
}

/// (len, mtime, content hash) per region file.
type RegionSnapshot = BTreeMap<String, (u64, std::time::SystemTime, u64)>;

fn snapshot_region_folder(root: &Path) -> RegionSnapshot {
    let region_dir = root.join("region");
    let mut snapshot = RegionSnapshot::new();
    let Ok(entries) = std::fs::read_dir(&region_dir) else {
        return snapshot;
    };
    for entry in entries.flatten() {
        let meta = entry.metadata().expect("metadata");
        if !meta.is_file() {
            continue;
        }
        let bytes = std::fs::read(entry.path()).expect("read region file");
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        snapshot.insert(
            entry.file_name().to_string_lossy().into_owned(),
            (meta.len(), meta.modified().expect("mtime"), hasher.finish()),
        );
    }
    snapshot
}

/// Files present in `before` that changed or disappeared in `after`, plus
/// files that appeared; returns (name, bytes involved) pairs.
fn snapshot_diff(before: &RegionSnapshot, after: &RegionSnapshot) -> Vec<(String, u64)> {
    let mut changed = Vec::new();
    for (name, meta) in before {
        match after.get(name) {
            Some(new_meta) if new_meta == meta => {}
            Some(new_meta) => changed.push((name.clone(), new_meta.0)),
            None => changed.push((name.clone(), meta.0)),
        }
    }
    for (name, meta) in after {
        if !before.contains_key(name) {
            changed.push((name.clone(), meta.0));
        }
    }
    changed
}

/// Opens a level over `root`, loads `positions` with tickets at `ticket_level`,
/// lets the pipeline settle, optionally exercises the explicit unload path,
/// then shuts down (flushes everything).
async fn run_level_cycle(
    root: &Path,
    positions: &[Vector2<i32>],
    ticket_level: i8,
    unload_before_shutdown: bool,
) {
    let level = Level::from_root_folder(
        &level_config(),
        root.to_path_buf(),
        SEED,
        Dimension::OVERWORLD,
        None,
    );
    level.world_portal.store(Arc::new(Some(
        Arc::new(BlockRegistry) as Arc<dyn WorldPortalExt>
    )));

    {
        let mut loading = level.chunk_loading.lock().unwrap();
        for pos in positions {
            loading.add_ticket(*pos, ticket_level);
        }
        loading.send_change();
    }

    // Proto-stage targets never become Full, so there is no chunk-listener
    // callback to wait on; flat generation of the small ring finishes well
    // within this settle window (unload batching runs on a 1 s cadence).
    tokio::time::sleep(Duration::from_millis(2500)).await;

    if unload_before_shutdown {
        {
            let mut loading = level.chunk_loading.lock().unwrap();
            for pos in positions {
                loading.remove_ticket(*pos, ticket_level);
            }
            loading.send_change();
        }
        level.should_unload.store(true, Relaxed);
        level.level_channel.notify();
        // > 1 s unload batching window plus io_write drain.
        tokio::time::sleep(Duration::from_millis(2500)).await;
    }

    level.shutdown().await;
}

/// Reading already-generated territory (same tickets, no modifications) must
/// leave every region file untouched: same length, same mtime, same bytes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn revisit_without_modification_writes_nothing() {
    let temp = temp_dir::TempDir::new().unwrap();
    let root = temp.path().to_path_buf();
    write_flat_world_settings(&root);

    let area = [Vector2::new(0, 0)];

    // Phase A: generate and flush.
    run_level_cycle(&root, &area, FEATURES_TICKET, false).await;
    // Phase B (warm-up revisit): absorbs chunks the phase-A shutdown may have
    // raced past; after this the on-disk state is complete.
    run_level_cycle(&root, &area, FEATURES_TICKET, false).await;

    let before = snapshot_region_folder(&root);
    assert!(
        !before.is_empty(),
        "expected region files after generation, found none"
    );

    // Phase C: pure revisit, explicit unload, then shutdown.
    run_level_cycle(&root, &area, FEATURES_TICKET, true).await;

    let after = snapshot_region_folder(&root);
    let changed = snapshot_diff(&before, &after);
    let total_bytes: u64 = changed.iter().map(|(_, b)| b).sum();
    assert!(
        changed.is_empty(),
        "revisit without modification rewrote {} region file(s), {} bytes: {:?}",
        changed.len(),
        total_bytes,
        changed
    );
}

/// Counter-test: a revisit that advances proto stages (weaker ticket first,
/// then a stronger one) must produce writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn revisit_with_stage_advancement_writes() {
    let temp = temp_dir::TempDir::new().unwrap();
    let root = temp.path().to_path_buf();
    write_flat_world_settings(&root);

    let area = [Vector2::new(0, 0)];

    // Phase A: only up to Carvers — the center stays a proto on disk.
    run_level_cycle(&root, &area, CARVERS_TICKET, false).await;
    let before = snapshot_region_folder(&root);
    assert!(
        !before.is_empty(),
        "expected region files after proto generation, found none"
    );

    // Phase B: request Features — protos must advance and be written.
    run_level_cycle(&root, &area, FEATURES_TICKET, false).await;

    let after = snapshot_region_folder(&root);
    assert!(
        !snapshot_diff(&before, &after).is_empty(),
        "stage advancement on revisit must rewrite the affected region"
    );
}
