use bevy::prelude::*;

use crate::{Config, Density, InstanceMaterialData, Particle};

use super::SpatialHash;

#[derive(Resource, Default, Clone)]
pub struct Measurements {
    pub delta_t: f32,
    pub tps: f32,
    pub p0_position: Vec3,
    pub p0_predicted_position: Vec3,
    pub p0_velocity: Vec2,
    pub p0_density: Density,
    pub p0_max_density_far: f32,
}

pub fn measurements_system(
    time: Res<Time>,
    mut measurements: ResMut<Measurements>,
    config: Res<Config>,
    particles_query: Query<&mut InstanceMaterialData, With<Particle>>,
    spatial_hash: Res<SpatialHash>,
) {
    if config.mark_sample_particle_neighbors && spatial_hash.first_entity_id != Entity::from_raw(0)
    {
        let data = particles_query.iter().next().unwrap();
        let instance = data[0];

        measurements.p0_position = instance.position.clone();
        measurements.p0_predicted_position = instance.predicted_position.clone();
        measurements.p0_velocity = instance.velocity.clone();
        measurements.p0_density = instance.density.clone();
        if instance.density.far > measurements.p0_max_density_far {
            measurements.p0_max_density_far = instance.density.far;
        }
    }

    if config.is_paused {
        return;
    }

    measurements.delta_t = time.delta_seconds() * config.time_scale;
    measurements.tps = 1. / measurements.delta_t;
}
