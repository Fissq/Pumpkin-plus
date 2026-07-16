//! Chunk NBT serialization must be deterministic: identical chunk data
//! serializes byte-identically across processes (and machines).
//!
//! ASLR seeds `std::collections::HashMap`'s RandomState differently per
//! process, so hash-order bugs are only observable BETWEEN processes: the
//! parent test computes a hash of freshly generated + serialized chunks,
//! then spawns this same test binary as a child (new ASLR layout) doing the
//! same, and compares the two hashes.

use pumpkin_data::BlockStateId;
use pumpkin_data::dimension::Dimension;
use pumpkin_util::world_seed::Seed;
use pumpkin_world::chunk_system::{Chunk, StagedChunkEnum, generate_single_chunk};
use pumpkin_world::generation::get_world_gen;
use pumpkin_world::world::WorldPortalExt;
use std::hash::{DefaultHasher, Hash, Hasher};

const SEED: Seed = Seed(42);
/// (0,0) exercises fluid ticks near water; (0,5) is dense with cave
/// vegetation patches — the two nondeterminism classes fixed on this branch.
const POSITIONS: [(i32, i32); 2] = [(0, 0), (0, 5)];
const HASH_PREFIX: &str = "NBT_DET_HASH=";

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

/// Generates the fixed chunk set and hashes the serialized NBT bytes.
async fn serialized_dataset_hash() -> u64 {
    let world_gen = get_world_gen(
        SEED,
        Dimension::OVERWORLD,
        false,
        Vec::new(),
        String::new(),
    );
    let registry = BlockRegistry;
    let mut hasher = DefaultHasher::new();

    for (x, z) in POSITIONS {
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
        use pumpkin_world::chunk::format::anvil::SingleChunkDataSerializer;
        let bytes = data.to_bytes().await.expect("serialize chunk");
        bytes.hash(&mut hasher);
    }

    hasher.finish()
}

/// Child entry point: run explicitly by the parent test below.
#[tokio::test]
#[ignore = "helper for serialization_is_deterministic_across_processes"]
async fn nbt_determinism_child() {
    println!("{HASH_PREFIX}{:x}", serialized_dataset_hash().await);
}

#[tokio::test]
async fn serialization_is_deterministic_across_processes() {
    let parent_hash = format!("{:x}", serialized_dataset_hash().await);

    let exe = std::env::current_exe().expect("current test binary path");
    let output = std::process::Command::new(exe)
        .args([
            "nbt_determinism_child",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .output()
        .expect("spawn child test process");
    assert!(
        output.status.success(),
        "child test process failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let child_hash = stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix(HASH_PREFIX))
        .unwrap_or_else(|| panic!("child produced no {HASH_PREFIX} line; stdout: {stdout}"));

    assert_eq!(
        parent_hash, child_hash,
        "chunk NBT serialization differs between processes (ASLR-dependent \
         container order leaked into generation or serialization)"
    );
}
