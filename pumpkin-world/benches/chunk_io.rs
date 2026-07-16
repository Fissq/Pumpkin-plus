//! Write-path IO benchmarks: `ChunkFileManager::save_chunks` driven directly,
//! bypassing the scheduler.
//!
//! Scenarios (each at write_in_place = false and true):
//! - hot cache: 64 chunks into one region, region serializer stays cached
//!   between iterations (via watchers — note that current code then defers
//!   the disk flush entirely, so this measures pure serialization);
//! - cold cache: current production behaviour — every save re-reads the
//!   region, rewrites it and evicts the serializer;
//! - pregen fill: empty region filled with 16 batches of 64 chunks
//!   (32×32 = 1024 total), save_chunks after every batch.
//!
//! Besides wall time, each scenario prints one machine-readable line:
//! `IO_BENCH_RESULT scenario=<name> written=<bytes> read=<bytes>`
//! (real numbers require `--features io-bench-counters`; otherwise `na`).
//! Useful payload sizes are printed as `IO_BENCH_PAYLOAD ...` lines.

use criterion::{Criterion, criterion_group, criterion_main};
use pumpkin_config::chunk::AnvilChunkConfig;
use pumpkin_data::BlockStateId;
use pumpkin_data::dimension::Dimension;
use pumpkin_util::math::vector2::Vector2;
use pumpkin_util::world_seed::Seed;
use pumpkin_world::chunk::ChunkData;
use pumpkin_world::chunk::format::anvil::{AnvilChunkFile, SingleChunkDataSerializer};
use pumpkin_world::chunk::io::file_manager::ChunkFileManager;
use pumpkin_world::chunk::io::{Dirtiable, FileIO, counters};
use pumpkin_world::chunk_system::{Chunk, StagedChunkEnum, generate_single_chunk};
use pumpkin_world::generation::generator::FlatLayer;
use pumpkin_world::generation::get_world_gen;
use pumpkin_world::level::{LevelFolder, SyncChunk};
use pumpkin_world::world::WorldPortalExt;
use std::path::Path;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

const SEED: Seed = Seed(42);

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

type Dataset = Vec<(Vector2<i32>, SyncChunk)>;
type Manager = ChunkFileManager<AnvilChunkFile<ChunkData>>;

fn generate_full_chunks(grid: i32, flat: bool) -> Dataset {
    let world_gen = if flat {
        get_world_gen(
            SEED,
            Dimension::OVERWORLD,
            true,
            vec![
                FlatLayer {
                    block: "minecraft:bedrock".into(),
                    height: 1,
                },
                FlatLayer {
                    block: "minecraft:dirt".into(),
                    height: 2,
                },
                FlatLayer {
                    block: "minecraft:grass_block".into(),
                    height: 1,
                },
            ],
            "minecraft:plains".into(),
        )
    } else {
        get_world_gen(SEED, Dimension::OVERWORLD, false, Vec::new(), String::new())
    };
    let registry = BlockRegistry;

    let mut chunks = Vec::with_capacity((grid * grid) as usize);
    for x in 0..grid {
        for z in 0..grid {
            let chunk = generate_single_chunk(
                &Dimension::OVERWORLD,
                0,
                &world_gen,
                &registry,
                x,
                z,
                StagedChunkEnum::Full,
            );
            let Chunk::Level(data) = chunk else {
                panic!("Full stage must yield a Level chunk");
            };
            chunks.push((Vector2::new(x, z), data));
        }
    }
    chunks
}

/// 64 vanilla-noise chunks (8×8, region r.0.0) — realistic payload sizes.
fn noise_chunks() -> &'static Dataset {
    static DATA: OnceLock<Dataset> = OnceLock::new();
    DATA.get_or_init(|| generate_full_chunks(8, false))
}

/// 1024 flat chunks (32×32, exactly one full region) for the pregen fill.
fn flat_chunks() -> &'static Dataset {
    static DATA: OnceLock<Dataset> = OnceLock::new();
    DATA.get_or_init(|| generate_full_chunks(32, true))
}

fn make_folder(root: &Path) -> LevelFolder {
    let region_folder = root.join("region");
    std::fs::create_dir_all(&region_folder).expect("create region dir");
    LevelFolder {
        root_folder: root.to_path_buf(),
        dim_folder: root.to_path_buf(),
        region_folder,
        entities_folder: root.join("entities"),
        poi_folder: root.join("poi"),
    }
}

fn make_manager(write_in_place: bool) -> Manager {
    ChunkFileManager::new(AnvilChunkConfig {
        compression: Default::default(),
        write_in_place,
    })
}

fn mark_all_dirty(data: &[(Vector2<i32>, SyncChunk)]) {
    for (_, chunk) in data {
        chunk.mark_dirty(true);
    }
}

async fn save(manager: &Manager, folder: &LevelFolder, data: &[(Vector2<i32>, SyncChunk)]) {
    manager
        .save_chunks(folder, data.to_vec())
        .await
        .expect("save_chunks failed");
}

fn print_io_result(scenario: &str) {
    match counters::snapshot() {
        Some((written, read)) => {
            println!("IO_BENCH_RESULT scenario={scenario} written={written} read={read}");
        }
        None => {
            println!(
                "IO_BENCH_RESULT scenario={scenario} written=na read=na \
                 (run with --features io-bench-counters)"
            );
        }
    }
}

/// Sum of the uncompressed serialized NBT sizes — the "useful" payload.
fn payload_nbt_bytes(rt: &Runtime, data: &[(Vector2<i32>, SyncChunk)]) -> u64 {
    rt.block_on(async {
        let mut total = 0u64;
        for (_, chunk) in data {
            total += chunk.to_bytes().await.expect("serialize chunk").len() as u64;
        }
        total
    })
}

fn region_file_size(folder: &LevelFolder) -> u64 {
    std::fs::read_dir(&folder.region_folder)
        .expect("read region dir")
        .flatten()
        .map(|e| e.metadata().expect("metadata").len())
        .sum()
}

fn wip_suffix(write_in_place: bool) -> &'static str {
    if write_in_place {
        "write_in_place"
    } else {
        "write_all"
    }
}

fn bench_hot_cache(c: &mut Criterion, rt: &Runtime, write_in_place: bool) {
    let data = noise_chunks();
    let positions: Vec<_> = data.iter().map(|(pos, _)| *pos).collect();
    let suffix = wip_suffix(write_in_place);

    // Counted single pass on a fresh state.
    {
        let temp = temp_dir::TempDir::new().expect("tempdir");
        let folder = make_folder(temp.path());
        let manager = make_manager(write_in_place);
        rt.block_on(manager.watch_chunks(&folder, &positions));
        counters::reset();
        mark_all_dirty(data);
        rt.block_on(save(&manager, &folder, data));
        print_io_result(&format!("hot_cache_{suffix}"));
    }

    c.bench_function(&format!("save_chunks_hot_cache/{suffix}"), |b| {
        let temp = temp_dir::TempDir::new().expect("tempdir");
        let folder = make_folder(temp.path());
        let manager = make_manager(write_in_place);
        rt.block_on(manager.watch_chunks(&folder, &positions));
        b.iter(|| {
            mark_all_dirty(data);
            rt.block_on(save(&manager, &folder, data));
        });
    });
}

fn bench_cold_cache(c: &mut Criterion, rt: &Runtime, write_in_place: bool) {
    let data = noise_chunks();
    let suffix = wip_suffix(write_in_place);

    // Counted single pass: pre-seed the region, then measure one
    // steady-state save (read region -> update -> write -> evict).
    {
        let temp = temp_dir::TempDir::new().expect("tempdir");
        let folder = make_folder(temp.path());
        let manager = make_manager(write_in_place);
        mark_all_dirty(data);
        rt.block_on(save(&manager, &folder, data));
        println!(
            "IO_BENCH_PAYLOAD dataset=noise64 nbt_bytes={} file_bytes={}",
            payload_nbt_bytes(rt, data),
            region_file_size(&folder)
        );
        counters::reset();
        mark_all_dirty(data);
        rt.block_on(save(&manager, &folder, data));
        print_io_result(&format!("cold_cache_{suffix}"));
    }

    c.bench_function(&format!("save_chunks_cold_cache/{suffix}"), |b| {
        let temp = temp_dir::TempDir::new().expect("tempdir");
        let folder = make_folder(temp.path());
        let manager = make_manager(write_in_place);
        mark_all_dirty(data);
        rt.block_on(save(&manager, &folder, data));
        b.iter(|| {
            mark_all_dirty(data);
            rt.block_on(save(&manager, &folder, data));
        });
    });
}

fn run_region_fill(rt: &Runtime, write_in_place: bool, folder: &LevelFolder) {
    let data = flat_chunks();
    let manager = make_manager(write_in_place);
    for batch in data.chunks(64) {
        mark_all_dirty(batch);
        rt.block_on(save(&manager, folder, batch));
    }
}

fn bench_region_fill(c: &mut Criterion, rt: &Runtime, write_in_place: bool) {
    let suffix = wip_suffix(write_in_place);

    // Counted pass: full fill of an empty region, totals for all 16 batches.
    {
        let temp = temp_dir::TempDir::new().expect("tempdir");
        let folder = make_folder(temp.path());
        counters::reset();
        run_region_fill(rt, write_in_place, &folder);
        print_io_result(&format!("region_fill_{suffix}"));
        println!(
            "IO_BENCH_PAYLOAD dataset=flat1024 nbt_bytes={} file_bytes={}",
            payload_nbt_bytes(rt, flat_chunks()),
            region_file_size(&folder)
        );
    }

    let mut group = c.benchmark_group("region_fill_pregen");
    group.sample_size(10);
    group.bench_function(suffix, |b| {
        b.iter(|| {
            let temp = temp_dir::TempDir::new().expect("tempdir");
            let folder = make_folder(temp.path());
            run_region_fill(rt, write_in_place, &folder);
        });
    });
    group.finish();
}

fn bench_chunk_io(c: &mut Criterion) {
    let rt = Runtime::new().expect("tokio runtime");

    for write_in_place in [false, true] {
        bench_hot_cache(c, &rt, write_in_place);
        bench_cold_cache(c, &rt, write_in_place);
        bench_region_fill(c, &rt, write_in_place);
    }
}

criterion_group!(benches, bench_chunk_io);
criterion_main!(benches);
