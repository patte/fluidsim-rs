use bevy::prelude::*;

use crate::{
    colors::{ColorSchemeCategoricalResource, GradientResource},
    get_cell_2d, hash_cell_2d, key_from_hash, Config, InstanceMaterialData, Particle,
    ParticleColorMode,
};

use super::{process_neighbors, SpatialHash};

pub fn color_system(
    config: Res<Config>,
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
    gradient_resource: Res<GradientResource>,
    color_scheme_categorical_resource: Res<ColorSchemeCategoricalResource>,
    spatial_hash: Res<SpatialHash>,
    mut gizmos: Gizmos,
) {
    let instance_index = 0;

    if config.mark_sample_particle_neighbors {
        particles_query.iter().for_each(|data| {
            let instance = data[instance_index as usize];

            let cell = get_cell_2d(instance.position.truncate(), config.smoothing_radius);
            let hash = hash_cell_2d(cell);
            let key = key_from_hash(hash, spatial_hash.indices.len() as u32);

            //println!("cell: {:?}  hash: {}  key: {}", cell, hash, key);

            let cell_color = color_scheme_categorical_resource
                .get_color_wrapped(&(key as usize))
                .clone()
                .with_alpha(0.6);

            // draw circle with smoothing_radius around particle0
            gizmos.circle_2d(
                instance.position.truncate(),
                config.smoothing_radius,
                Color::srgba(1., 1., 1., 0.3),
            );

            process_neighbors(
                &instance.position,
                &spatial_hash,
                &config,
                |neighbor_entity_id| {
                    let instance2 = data[neighbor_entity_id as usize];

                    let offset = instance2.position - instance.position;
                    let sqrt_dst = offset.length_squared();

                    // skip if too far
                    if sqrt_dst > config.smoothing_radius.powf(2.0) {
                        return;
                    }

                    // draw line to each neighbor
                    gizmos.line_2d(
                        instance2.position.truncate(),
                        instance.position.truncate(),
                        cell_color,
                    );
                },
                Some(instance_index),
                false,
            );
        });
    }

    if config.is_paused {
        return;
    }

    particles_query.iter_mut().for_each(|mut data| {
        for i in 0..data.len() {
            let instance = data[i];

            if config.particle_color_mode == ParticleColorMode::Velocity {
                let speed_normalized = instance.velocity.length() / config.max_velocity_for_color;
                //println!("speed_normalized {}", speed_normalized);
                data[i].color = gradient_resource
                    .get_gradient_color(&speed_normalized)
                    .to_srgba()
                    .to_f32_array();
            } else if config.particle_color_mode == ParticleColorMode::Density {
                let density_normalized = instance.density.far / config.max_density_for_color;
                //println!("density_normalized {}", density_normalized);
                data[i].color = gradient_resource
                    .get_gradient_color(&density_normalized)
                    .to_srgba()
                    .to_f32_array();
            } else if config.particle_color_mode == ParticleColorMode::CellKey
                && spatial_hash.indices.len() > 0
            {
                let cell = get_cell_2d(instance.position.truncate(), config.smoothing_radius);
                let hash = hash_cell_2d(cell);
                let key = key_from_hash(hash, spatial_hash.indices.len() as u32);
                let wrapped_color_index =
                    color_scheme_categorical_resource.get_wrapped_index(&(key as usize));
                data[i].color = color_scheme_categorical_resource
                    .get_color_wrapped(&wrapped_color_index)
                    .to_srgba()
                    .to_f32_array();
            } else if config.particle_color_mode == ParticleColorMode::Blue {
                data[i].color = [0.0, 0.0, 1.0, 1.0];
            }
        }
    });
}
