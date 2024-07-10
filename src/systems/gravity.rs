use bevy::prelude::*;

use crate::{Config, InstanceMaterialData, InteractionInputs, Particle};

pub fn gravity_system(
    time: Res<Time>,
    config: Res<Config>,
    mut particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
    interaction_inputs: Res<InteractionInputs>,
) {
    if config.is_paused {
        return;
    }

    let delta_t = time.delta_seconds() * config.time_scale;

    particles_query.par_iter_mut().for_each(|mut data| {
        for instance in data.iter_mut() {
            let mut acceleration = config.gravity;

            if interaction_inputs.strength != 0. {
                let input_point_offset = interaction_inputs.point.unwrap() - instance.position.xy();
                let sqr_dst = input_point_offset.length_squared();
                if sqr_dst < config.interaction_input_radius.powf(2.) {
                    let dst = sqr_dst.sqrt();
                    let edge_t = dst / config.interaction_input_radius;
                    let centre_t = 1. - edge_t;
                    let dir_to_centre = input_point_offset / dst;

                    let gravity_weight = 1. - (centre_t * (interaction_inputs.strength / 10.));

                    acceleration = acceleration * gravity_weight
                        + dir_to_centre * centre_t * interaction_inputs.strength;
                    acceleration -= instance.velocity * centre_t;
                }
            }

            instance.velocity += acceleration * delta_t;
            instance.predicted_position = instance.position
                + instance.velocity.extend(0.) * (delta_t as f32 * config.prediction_time_scale);
        }
    });
}
