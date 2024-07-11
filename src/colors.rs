use bevy::{
    color::palettes::basic::BLUE, color::palettes::basic::GREEN, color::palettes::basic::RED,
    prelude::*,
};

use bevy_internal::prelude::Vec4;

use crate::vendor::Gradient;

#[derive(Resource)]
pub struct GradientResource {
    colors: Vec<Color>,
}

#[derive(Resource)]
pub struct ColorSchemeCategoricalResource {
    colors: Vec<Color>,
}

impl GradientResource {
    pub fn new() -> Self {
        let mut gradient = Gradient::new();
        gradient.add_key(0.0, BLUE.to_vec4());
        gradient.add_key(0.5, GREEN.to_vec4());
        gradient.add_key(1.0, RED.to_vec4());

        let num_precomputed_colors = 100;
        let colors = (0..num_precomputed_colors)
            .map(|i| {
                let gradient_point: Vec4 =
                    gradient.sample(i as f32 / num_precomputed_colors as f32);
                Color::srgba(
                    gradient_point.x,
                    gradient_point.y,
                    gradient_point.z,
                    gradient_point.w,
                )
            })
            .collect::<Vec<Color>>();

        Self { colors }
    }

    pub fn get_gradient_color(&self, ratio: &f32) -> Color {
        return self.colors[(ratio.max(0.).min(1.) * (self.colors.len() - 1) as f32) as usize]
            .clone();
    }
}

// set2 of https://observablehq.com/@d3/color-schemes
impl ColorSchemeCategoricalResource {
    pub fn new() -> Self {
        Self {
            colors: vec![
                Srgba::hex("66c2a5").unwrap().into(),
                Srgba::hex("fc8d62").unwrap().into(),
                Srgba::hex("8da0cb").unwrap().into(),
                Srgba::hex("e78ac3").unwrap().into(),
                Srgba::hex("a6d854").unwrap().into(),
                Srgba::hex("ffd92f").unwrap().into(),
                Srgba::hex("e5c494").unwrap().into(),
                Srgba::hex("b3b3b3").unwrap().into(),
            ],
        }
    }

    pub fn get_color_wrapped(&self, index: &usize) -> Color {
        return self.colors[index % self.colors.len()];
    }

    pub fn get_wrapped_index(&self, index: &usize) -> usize {
        return index % self.colors.len();
    }
}
