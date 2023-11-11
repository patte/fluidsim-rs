use bevy::math::Vec2;

static OFFSETS_2D: [Vec2; 9] = [
    Vec2::new(-1., 1.),
    Vec2::new(0., 1.),
    Vec2::new(1., 1.),
    Vec2::new(-1., 0.),
    Vec2::new(0., 0.),
    Vec2::new(1., 0.),
    Vec2::new(-1., -1.),
    Vec2::new(0., -1.),
    Vec2::new(1., -1.),
];

// Constants used for hashing
const HASH_K1: u32 = 15823;
const HASH_K2: u32 = 9737333;

// Convert floating point position into an integer cell coordinate
pub fn get_cell_2d(position: Vec2, radius: f32) -> Vec2 {
    // move negative values into cell
    let position = position + Vec2::new(500., 500.);
    (position / radius).floor()
}

// Hash cell coordinate to a single unsigned integer
pub fn hash_cell_2d(cell: Vec2) -> u32 {
    let a = (cell.x as u32) * HASH_K1;
    let b = (cell.y as u32) * HASH_K2;
    a + b
}

pub fn key_from_hash(hash: u32, table_size: u32) -> u32 {
    hash % table_size
}
