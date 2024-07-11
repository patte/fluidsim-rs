use bevy::prelude::*;

use crate::{Config, Density, InstanceMaterialData, Particle};

#[derive(Resource, Default, Clone)]
pub struct Measurements {
    pub delta_t: f32,
    pub tps: f32,
    pub p0_position: Vec3,
    pub p0_predicted_position: Vec3,
    pub p0_velocity: Vec2,
    p0_last_100_velocities: Vec<Vec2>,
    pub p0_velocity_avg: Vec2,
    pub p0_velocity_max: Vec2,
    pub p0_density: Density,
    pub p0_max_density_far: f32,
}

pub fn measurements_system(
    time: Res<Time>,
    mut measurements: ResMut<Measurements>,
    config: Res<Config>,
    particles_query: Query<&InstanceMaterialData, With<Particle>>,
) {
    let instance_index = 0;

    if config.mark_sample_particle_neighbors && instance_index != 0 {
        let data = particles_query.iter().next();

        if let Some(data) = data {
            let instance = data[instance_index as usize];

            measurements.p0_position = instance.position.clone();
            measurements.p0_predicted_position = instance.predicted_position.clone();
            measurements.p0_velocity = instance.velocity.clone();
            measurements.p0_density = instance.density.clone();
            if instance.density.far > measurements.p0_max_density_far {
                measurements.p0_max_density_far = instance.density.far;
            }

            measurements
                .p0_last_100_velocities
                .push(instance.velocity.clone());
            if measurements.p0_last_100_velocities.len() > 100 {
                measurements.p0_last_100_velocities.remove(0);
            }
            measurements.p0_velocity_avg = measurements
                .p0_last_100_velocities
                .iter()
                .fold(Vec2::ZERO, |acc, v| acc + *v)
                / measurements.p0_last_100_velocities.len() as f32;

            if measurements.p0_velocity_max.length_squared() < instance.velocity.length_squared() {
                measurements.p0_velocity_max = instance.velocity.clone();
            }
        }
    }

    if config.is_paused {
        return;
    }

    measurements.delta_t = time.delta_seconds() * config.time_scale;
    measurements.tps = 1. / measurements.delta_t;
}
