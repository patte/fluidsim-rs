use std::f32::consts::PI;

static SMOOTHING_RADIUS: f32 = 0.35;
static SMOOTHING_RADIUS_POW_2: f32 = SMOOTHING_RADIUS * SMOOTHING_RADIUS;
static SMOOTHING_RADIUS_POW_3: f32 = SMOOTHING_RADIUS_POW_2 * SMOOTHING_RADIUS;
static SMOOTHING_RADIUS_POW_4: f32 = SMOOTHING_RADIUS_POW_3 * SMOOTHING_RADIUS;
static SMOOTHING_RADIUS_POW_5: f32 = SMOOTHING_RADIUS_POW_4 * SMOOTHING_RADIUS;

static SPIKY_KERNEL_POW_2_FACTOR: f32 = 6.0 / (PI * SMOOTHING_RADIUS_POW_4);
static SPIKY_KERNEL_POW_3_FACTOR: f32 = 10.0 / (PI * SMOOTHING_RADIUS_POW_5);
static DERIVATIVE_SPIKY_POW_2_FACTOR: f32 = 12.0 / (PI * SMOOTHING_RADIUS_POW_4);
static DERIVATIVE_SPIKY_POW_3_FACTOR: f32 = 30.0 / (PI * SMOOTHING_RADIUS_POW_5);

pub fn spiky_kernel_pow_2(radius: &f32, distance: &f32) -> f32 {
    if distance >= radius {
        return 0.;
    }
    let fac = if radius == &SMOOTHING_RADIUS {
        SPIKY_KERNEL_POW_2_FACTOR
    } else {
        6.0 / (PI * radius.powf(4.0))
    };
    let v = radius - distance;
    return v * v * fac;
}

pub fn spiky_kernel_pow_3(radius: &f32, distance: &f32) -> f32 {
    if distance >= radius {
        return 0.;
    }
    let fac = if radius == &SMOOTHING_RADIUS {
        SPIKY_KERNEL_POW_3_FACTOR
    } else {
        10.0 / (PI * radius.powf(5.0))
    };
    let v = radius - distance;
    return v * v * v * fac;
}

pub fn derivative_spiky_pow_2(radius: &f32, distance: &f32) -> f32 {
    if distance > radius {
        return 0.;
    }
    let fac = if radius == &SMOOTHING_RADIUS {
        DERIVATIVE_SPIKY_POW_2_FACTOR
    } else {
        12.0 / (PI * radius.powf(4.0))
    };
    let v = radius - distance;
    return -v * fac;
}

pub fn derivative_spiky_pow_3(radius: &f32, distance: &f32) -> f32 {
    if distance > radius {
        return 0.;
    }
    let fac = if radius == &SMOOTHING_RADIUS {
        DERIVATIVE_SPIKY_POW_3_FACTOR
    } else {
        30.0 / (PI * radius.powf(5.0))
    };
    let v = radius - distance;
    return -v * v * fac;
}
