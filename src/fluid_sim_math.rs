use std::f32::consts::PI;

pub fn spiky_kernel_pow_2(radius: &f32, distance: &f32) -> f32 {
    if distance >= radius {
        return 0.;
    }
    let v = radius - distance;
    return v * v * (6.0 / (PI * radius.powf(4.0)));
}

pub fn spiky_kernel_pow_3(radius: &f32, distance: &f32) -> f32 {
    if distance >= radius {
        return 0.;
    }
    let v = radius - distance;
    return v * v * v * (10.0 / (PI * radius.powf(5.0)));
}

pub fn derivative_spiky_pow_2(radius: &f32, distance: &f32) -> f32 {
    if distance > radius {
        return 0.;
    }
    let v = radius - distance;
    return -v * (12.0 / (PI * radius.powf(4.0)));
}

pub fn derivative_spiky_pow_3(radius: &f32, distance: &f32) -> f32 {
    if distance > radius {
        return 0.;
    }
    let v = radius - distance;
    return -v * v * (30.0 / (PI * radius.powf(5.0)));
}
