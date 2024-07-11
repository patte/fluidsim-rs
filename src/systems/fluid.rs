use crate::{
    derivative_spiky_pow_2, derivative_spiky_pow_3, get_cell_2d, hash_cell_2d, key_from_hash,
    spiky_kernel_pow_2, spiky_kernel_pow_3, Config, InstanceMaterialData, Particle, OFFSETS_2D,
};
use bevy::prelude::*;

#[derive(Clone, Debug)]
pub struct SpatialIndex {
    key: u32,
    hash: u32,
    instance_index: u32,
}

impl Default for SpatialIndex {
    fn default() -> Self {
        Self {
            key: u32::MAX,
            hash: u32::MAX,
            instance_index: 0,
        }
    }
}

#[derive(Resource)]
pub struct SpatialHash {
    pub indices: Vec<SpatialIndex>,
    pub offsets: Vec<usize>,
}

impl Default for SpatialHash {
    fn default() -> Self {
        Self {
            indices: Vec::<SpatialIndex>::new(),
            offsets: Vec::new(),
        }
    }
}

pub fn update_spatial_hash_system(
    mut spatial_hash: ResMut<SpatialHash>,
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
    config: Res<Config>,
) {
    if config.is_paused {
        return;
    }

    let num_particles = particles_query
        .iter_mut()
        .fold(0, |acc, data| acc + data.len());

    // resize
    if num_particles > spatial_hash.indices.len() {
        spatial_hash.indices.resize(
            num_particles,
            SpatialIndex {
                key: u32::MAX,
                hash: u32::MAX,
                instance_index: 0,
            },
        );
        spatial_hash.offsets.resize(num_particles, usize::MAX);
        println!("spatial_hash.indices.len(): {}", spatial_hash.indices.len());
    }

    // new indices
    let mut new_indices: Vec<SpatialIndex> = Vec::new();

    for data in particles_query.iter_mut() {
        for i in 0..data.len() {
            let instance = data[i];

            let cell = get_cell_2d(
                instance.predicted_position.truncate(),
                config.smoothing_radius,
            );
            let hash = hash_cell_2d(cell);
            let key = key_from_hash(hash, spatial_hash.indices.len() as u32);
            new_indices.push(SpatialIndex {
                key,
                hash,
                instance_index: i as u32,
            });
        }
    }

    // sort by key
    new_indices.sort_by(|a, b| a.key.partial_cmp(&b.key).unwrap());

    // reset offsets
    spatial_hash
        .offsets
        .iter_mut()
        .for_each(|offset| *offset = usize::MAX);

    // set spatial_hash.offsets to the first index of each hash
    let mut last_key = u32::MAX;
    new_indices.iter().enumerate().for_each(|(i, index)| {
        if index.key != last_key {
            spatial_hash.offsets[index.key as usize] = i;
            last_key = index.key;
        }
    });

    spatial_hash.indices = new_indices;

    /*
    println!("spatial_hash.indices.len(): {}", spatial_hash.indices.len());
    println!("spatial_hash.offsets.len(): {}", spatial_hash.offsets.len());
    println!(
        "spatial_hash.first_instance_index: {:?}",
        spatial_hash.first_instance_index
    );
    println!("spatial_hash: {:?}", spatial_hash.indices);*/
}

pub fn calculate_density_system(
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }

    particles_query.iter_mut().for_each(|mut data| {
        for i in 0..data.len() {
            let instance = data[i];
            let mut density_sum = 0.;
            let mut density_near_sum = 0.;

            process_neighbors(
                &instance.predicted_position,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let neighbor_instance = data[neighbor_entity_id as usize];

                    let sqrt_dst = (neighbor_instance.predicted_position
                        - instance.predicted_position)
                        .length_squared();

                    // skip if too far
                    if sqrt_dst > config.smoothing_radius.powf(2.0) {
                        return;
                    }

                    let distance = sqrt_dst.sqrt();

                    density_sum += spiky_kernel_pow_2(&config.smoothing_radius, &distance);
                    density_near_sum += spiky_kernel_pow_3(&config.smoothing_radius, &distance);
                },
                None, // include self
                false,
            );

            data[i].density.far = density_sum;
            data[i].density.near = density_near_sum;
        }
    });
}

pub fn pressure_force_system(
    time: Res<Time>,
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }

    let delta_t = time.delta_seconds() * config.time_scale;

    let pressure_from_density = |density: f32| -> f32 {
        return (density - config.target_density) * config.pressure_multiplier;
    };

    let near_pressure_from_density = |density: &f32| -> f32 {
        return density * config.near_pressure_multiplier;
    };

    let random_direction = Vec2::new(0., 1.);

    particles_query.iter_mut().for_each(|mut data| {
        for i in 0..data.len() {
            let instance = data[i];
            let mut sum_pressure_force = Vec2::ZERO;
            let pressure = pressure_from_density(instance.density.far);
            let near_pressure = near_pressure_from_density(&instance.density.near);

            process_neighbors(
                &instance.predicted_position,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let neighbor_instance = data[neighbor_entity_id as usize];

                    let offset = neighbor_instance.predicted_position - instance.predicted_position;
                    let sqrt_dst = offset.length_squared();

                    // skip if too far
                    if sqrt_dst > config.smoothing_radius.powf(2.0) {
                        return;
                    }

                    let distance = sqrt_dst.sqrt();
                    let direction = if distance > 0. {
                        offset.xy() / distance
                    } else {
                        random_direction
                    };

                    let shared_pressure =
                        (pressure + pressure_from_density(neighbor_instance.density.far)) * 0.5;
                    let shared_pressure_near = (near_pressure
                        + near_pressure_from_density(&neighbor_instance.density.near))
                        * 0.5;

                    sum_pressure_force += direction
                        * derivative_spiky_pow_2(&config.smoothing_radius, &distance)
                        * shared_pressure
                        / neighbor_instance.density.far;

                    sum_pressure_force += direction
                        * derivative_spiky_pow_3(&config.smoothing_radius, &distance)
                        * shared_pressure_near
                        / neighbor_instance.density.near;
                },
                Some(instance.entity_id), // exclude self
                false,
            );

            let acceleration = sum_pressure_force / instance.density.far;
            if acceleration.x.is_nan() || acceleration.y.is_nan() {
                continue;
            }

            data[i].velocity += acceleration * delta_t;

            if data[i].velocity.x.is_nan() || data[i].velocity.y.is_nan() {
                println!(
                    "acceleration: {:?} delta_t: {} acceleration_plus: {}",
                    acceleration,
                    delta_t,
                    acceleration * delta_t
                );
                panic!("data[i].velocity.x.is_nan() || data[i].velocity.y.is_nan()");
            }
        }
    });
}

pub fn process_neighbors<F>(
    me_position: &Vec3,
    spatial_hash: &SpatialHash,
    config: &Config,
    mut process: F,
    skip_entity_id: Option<u32>,
    log: bool,
) where
    F: FnMut(u32),
{
    let original_cell = get_cell_2d(me_position.truncate(), config.smoothing_radius);
    let original_hash = hash_cell_2d(original_cell);
    let original_key = key_from_hash(original_hash, spatial_hash.indices.len() as u32);

    if log {
        println!(
            "original_cell: {:?}  hash: {}  key: {}",
            original_cell, original_hash, original_key
        );
    }

    for offset in OFFSETS_2D.iter() {
        let cell = original_cell + *offset;
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);

        if log && *offset == Vec2::new(0., 0.) {
            println!(
                "  offset: {} cell: {:?}  hash: {}  key: {}",
                offset, cell, hash, key
            );
        }

        if let Some(&start_index) = spatial_hash.offsets.get(key as usize) {
            // a great number of time has been spent here on a off by one error spatial_hash.indices.len()-1
            for i in start_index as usize..spatial_hash.indices.len() {
                let index_data = spatial_hash.indices.get(i as usize).unwrap();
                if log {
                    println!(
                        "    key {} start_index: {} i: {} index_data.index {:?}",
                        key, start_index, i, index_data.instance_index
                    );
                }

                if index_data.key != key {
                    break;
                }
                if index_data.hash != hash {
                    continue;
                }

                if skip_entity_id.is_some() && skip_entity_id.unwrap() == index_data.instance_index
                {
                    continue;
                }

                process(index_data.instance_index);
            }
        }
    }
}
