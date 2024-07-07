use bevy::prelude::*;

use crate::{Config, InteractionInputs, Particle, PredictedPosition, Velocity};

pub fn gravity_system(
    time: Res<Time>,
    config: Res<Config>,
    mut particles_query: Query<(&Transform, &mut PredictedPosition, &mut Velocity), With<Particle>>,
    interaction_inputs: Res<InteractionInputs>,
) {
    if config.is_paused {
        return;
    }

    let delta_t = time.delta_seconds() * config.time_scale;

    particles_query
        .par_iter_mut()
        .for_each(|(transform, mut predicted_position, mut velocity)| {
            let mut acceleration = config.gravity;

            if interaction_inputs.strength != 0. {
                let input_point_offset =
                    interaction_inputs.point.unwrap() - transform.translation.xy();
                let sqr_dst = input_point_offset.length_squared();
                if sqr_dst < config.interaction_input_radius.powf(2.) {
                    let dst = sqr_dst.sqrt();
                    let edge_t = dst / config.interaction_input_radius;
                    let centre_t = 1. - edge_t;
                    let dir_to_centre = input_point_offset / dst;

                    let gravity_weight = 1. - (centre_t * (interaction_inputs.strength / 10.));

                    acceleration = acceleration * gravity_weight
                        + dir_to_centre * centre_t * interaction_inputs.strength;
                    acceleration -= velocity.0 * centre_t;
                }
            }

            velocity.0 += acceleration * delta_t;
            predicted_position.0 = transform.translation
                + velocity.0.extend(0.) * (delta_t as f32 * config.prediction_time_scale);
        });
}
