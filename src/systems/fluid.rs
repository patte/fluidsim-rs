use crate::{
    derivative_spiky_pow_2, derivative_spiky_pow_3, get_cell_2d, hash_cell_2d, key_from_hash,
    spiky_kernel_pow_2, spiky_kernel_pow_3, Config, Density, Particle, PredictedPosition,
    SpatialHash, SpatialIndex, Velocity, OFFSETS_2D,
};
use bevy::prelude::*;

pub fn update_spatial_hash_system(
    mut spatial_hash: ResMut<SpatialHash>,
    mut particles_query: Query<(Entity, &PredictedPosition), With<Particle>>,
    config: Res<Config>,
) {
    if config.is_paused {
        return;
    }

    let num_particles = particles_query.iter_mut().len();

    // resize
    if num_particles > spatial_hash.indices.len() {
        spatial_hash.indices.resize(
            num_particles,
            SpatialIndex {
                key: u32::MAX,
                hash: u32::MAX,
                entity_id: Entity::from_raw(0),
            },
        );
        spatial_hash.offsets.resize(num_particles, usize::MAX);
        println!("spatial_hash.indices.len(): {}", spatial_hash.indices.len());
    }

    // new indices
    let mut new_indices: Vec<SpatialIndex> = Vec::new();

    // remember first entity id
    let mut first_entity_id = Entity::from_raw(0);

    for (entity_id, predicted_position) in particles_query.iter_mut() {
        let cell = get_cell_2d(predicted_position.0.truncate(), config.smoothing_radius);
        let hash = hash_cell_2d(cell);
        let key = key_from_hash(hash, spatial_hash.indices.len() as u32);
        new_indices.push(SpatialIndex {
            key,
            hash,
            entity_id,
        });
        if first_entity_id == Entity::from_raw(0) {
            first_entity_id = entity_id;
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

    spatial_hash.first_entity_id = first_entity_id;
}

pub fn calculate_density_system(
    mut particles_query: Query<(&PredictedPosition, &mut Density), With<Density>>,
    particles_query_inner: Query<&PredictedPosition, With<Particle>>,
    config: Res<Config>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.is_paused {
        return;
    }

    particles_query
        .par_iter_mut()
        .for_each(|(predicted_position, mut density)| {
            let mut density_sum = 0.;
            let mut density_near_sum = 0.;

            process_neighbors(
                &predicted_position.0,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let neighbor_predicted_position =
                        particles_query_inner.get(neighbor_entity_id).unwrap();

                    let sqrt_dst =
                        (neighbor_predicted_position.0 - predicted_position.0).length_squared();

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

            density.far = density_sum;
            density.near = density_near_sum;
        });
}

pub fn pressure_force_system(
    time: Res<Time>,
    mut particles_query: Query<
        (Entity, &PredictedPosition, &mut Velocity, &Density),
        With<Particle>,
    >,
    particles_query_inner: Query<(&PredictedPosition, &Density), With<Particle>>,
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

    //let mut rng = thread_rng();
    //let random_direction = Vec2::new(rng.gen_range(-1. ..1.), rng.gen_range(-1. ..1.)).normalize();
    let random_direction = Vec2::new(0., 1.);

    particles_query.par_iter_mut().for_each(
        |(entity_id, predicted_position, mut velocity, density)| {
            let mut sum_pressure_force = Vec2::ZERO;
            let pressure = pressure_from_density(density.far);
            let near_pressure = near_pressure_from_density(&density.near);

            process_neighbors(
                &predicted_position.0,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let (predicted_position2, density2) =
                        particles_query_inner.get(neighbor_entity_id).unwrap();

                    let offset = predicted_position2.0 - predicted_position.0;
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

                    let shared_pressure = (pressure + pressure_from_density(density2.far)) * 0.5;
                    let shared_pressure_near =
                        (near_pressure + near_pressure_from_density(&density2.near)) * 0.5;

                    sum_pressure_force += direction
                        * derivative_spiky_pow_2(&config.smoothing_radius, &distance)
                        * shared_pressure
                        / density2.far;

                    sum_pressure_force += direction
                        * derivative_spiky_pow_3(&config.smoothing_radius, &distance)
                        * shared_pressure_near
                        / density2.near;
                },
                Some(entity_id), // exclude self
                false,
            );

            let acceleration = sum_pressure_force / density.far;

            velocity.0 += acceleration * delta_t;
        },
    );
}

pub fn process_neighbors<F>(
    me_position: &Vec3,
    spatial_hash: &SpatialHash,
    config: &Config,
    mut process: F,
    skip_entity_id: Option<Entity>,
    log: bool,
) where
    F: FnMut(Entity),
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
                        key, start_index, i, index_data.entity_id
                    );
                }

                if index_data.key != key {
                    break;
                }
                if index_data.hash != hash {
                    continue;
                }

                if skip_entity_id.is_some() && skip_entity_id.unwrap() == index_data.entity_id {
                    continue;
                }

                process(index_data.entity_id);
            }
        }
    }
}
