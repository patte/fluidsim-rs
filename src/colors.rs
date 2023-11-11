use bevy::prelude::*;

use bevy_hanabi::Gradient;
use bevy_internal::{prelude::Vec4, sprite::ColorMaterial};

use super::GradientResource;

impl GradientResource {
    pub fn new() -> Self {
        let mut gradient = Gradient::new();
        gradient.add_key(0.0, Color::BLUE.into());
        gradient.add_key(1.0, Color::RED.into());

        Self {
            gradient,
            precomputed_materials: vec![Handle::default(); 100],
        }
    }

    pub fn precompute_materials(&mut self, materials: &mut ResMut<Assets<ColorMaterial>>) {
        let num_precomputed_colors = 100;
        for i in 0..num_precomputed_colors {
            let gradient_point: Vec4 = self
                .gradient
                .sample(i as f32 / num_precomputed_colors as f32);
            let color = Color::rgba(
                gradient_point.x,
                gradient_point.y,
                gradient_point.z,
                gradient_point.w,
            );
            self.precomputed_materials[i] = materials.add(color.into());
        }
    }

    pub fn get_gradient_color_material(&self, ratio: &f32) -> Handle<ColorMaterial> {
        return self.precomputed_materials
            [(ratio.max(0.).min(1.) * (self.precomputed_materials.len() - 1) as f32) as usize]
            .clone();
    }
}

// https://observablehq.com/@d3/color-schemes
// set2
//["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3"]
use super::ColorSchemeCategoricalResource;
impl ColorSchemeCategoricalResource {
    fn num_colors() -> usize {
        8
    }
    pub fn new() -> Self {
        Self {
            precomputed_materials: vec![Handle::default(); Self::num_colors()],
        }
    }

    pub fn precompute_materials(&mut self, materials: &mut ResMut<Assets<ColorMaterial>>) {
        let colors = vec![
            Color::hex("66c2a5").unwrap(),
            Color::hex("fc8d62").unwrap(),
            Color::hex("8da0cb").unwrap(),
            Color::hex("e78ac3").unwrap(),
            Color::hex("a6d854").unwrap(),
            Color::hex("ffd92f").unwrap(),
            Color::hex("e5c494").unwrap(),
            Color::hex("b3b3b3").unwrap(),
        ];
        for i in 0..colors.len() {
            self.precomputed_materials[i] = materials.add(colors[i].into());
        }
    }

    pub fn get_color_material_wrapped(&self, index: &usize) -> Handle<ColorMaterial> {
        return self.precomputed_materials[index % self.precomputed_materials.len()].clone();
    }

    pub fn get_wrapped_index(&self, index: &usize) -> usize {
        return index % Self::num_colors();
    }
}
