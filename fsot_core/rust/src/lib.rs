//! FSOT 2.1 scalar engine — Rust path.
//! Formula mirrors archive `vendor/fsot_compute.py` `compute_scalar`.
//! Seeds only: π, e, φ, γ, G(Catalan). No free parameters.

#![allow(non_snake_case)]

pub const PI: f64 = std::f64::consts::PI;
pub const E: f64 = std::f64::consts::E;
pub const PHI: f64 = 1.618_033_988_749_895;
pub const GAMMA: f64 = 0.577_215_664_901_532_9;
pub const G_CAT: f64 = 0.915_965_594_177_219_0;

#[inline]
pub fn alpha() -> f64 {
    PI.ln() / (E * PHI.powi(13))
}
#[inline]
pub fn psi_con() -> f64 {
    1.0 - (-1.0_f64).exp()
}
#[inline]
pub fn eta_eff() -> f64 {
    1.0 / (PI - 1.0)
}
#[inline]
pub fn beta() -> f64 {
    1.0 / (PI.powf(PI) + (E - 1.0)).exp()
}
#[inline]
pub fn gamma_c() -> f64 {
    -2.0_f64.ln() / PHI
}
#[inline]
pub fn omega() -> f64 {
    (PI / E).sin() * 2.0_f64.sqrt()
}
#[inline]
pub fn theta_s() -> f64 {
    (psi_con() * eta_eff()).sin()
}
#[inline]
pub fn poof() -> f64 {
    ((-PI.ln() / E) / (eta_eff() * PHI.ln())).exp()
}
#[inline]
pub fn c_eff() -> f64 {
    (1.0 - poof() * theta_s().sin()) * (1.0 + 0.01 * G_CAT / (PI * PHI))
}
#[inline]
pub fn a_bleed() -> f64 {
    (PI / E).sin() * PHI / 2.0_f64.sqrt()
}
#[inline]
pub fn p_var() -> f64 {
    -(theta_s() + PI).cos()
}
#[inline]
pub fn b_in() -> f64 {
    c_eff() * (1.0 - theta_s().sin() / PHI)
}
#[inline]
pub fn a_in() -> f64 {
    a_bleed() * (1.0 + theta_s().cos() / PHI)
}
#[inline]
pub fn suction() -> f64 {
    poof() * (-(theta_s() - PI).cos())
}
#[inline]
pub fn chaos() -> f64 {
    gamma_c() / omega()
}
#[inline]
pub fn p_new() -> f64 {
    (GAMMA / E) * 2.0_f64.sqrt()
}
#[inline]
pub fn c_factor() -> f64 {
    c_eff() * p_new()
}
#[inline]
pub fn K() -> f64 {
    PHI * (GAMMA / E) * 2.0_f64.sqrt() / PI.ln() * 0.99
}

#[derive(Clone, Debug)]
pub struct ScalarInput {
    pub N: f64,
    pub P: f64,
    pub D_eff: f64,
    pub psi_con_v: f64,
    pub delta_psi: f64,
    pub recent_hits: f64,
    pub rho: f64,
    pub observed: bool,
    pub delta_theta: f64,
    pub scale: f64,
    pub amplitude: f64,
    pub trend_bias: f64,
}

impl Default for ScalarInput {
    fn default() -> Self {
        Self {
            N: 1.0,
            P: 1.0,
            D_eff: 25.0,
            psi_con_v: 0.0,
            delta_psi: 1.0,
            recent_hits: 0.0,
            rho: 1.0,
            observed: false,
            delta_theta: 1.0,
            scale: 1.0,
            amplitude: 1.0,
            trend_bias: 0.0,
        }
    }
}

/// S = K · (T1 + T2 + T3)
pub fn compute_scalar(s: &ScalarInput) -> f64 {
    let N = s.N;
    let P = s.P;
    let D = s.D_eff;
    let dp = s.delta_psi;
    let dt = s.delta_theta;
    let hits = s.recent_hits;
    let a = alpha();
    let psi = if s.psi_con_v != 0.0 {
        s.psi_con_v
    } else {
        psi_con()
    };
    let eta = eta_eff();
    let ce = c_eff();
    let pn = p_new();
    let bin = b_in();
    let ain = a_in();
    let abl = a_bleed();
    let pv = p_var();
    let pf = poof();
    let su = suction();
    let ch = chaos();
    let bt = beta();
    let th = theta_s();
    let cf = c_factor();

    let growth = (a * (1.0 - hits / N) * GAMMA / PHI).exp();
    let base = (N * P / D.sqrt())
        * ((psi + dp) / eta).cos()
        * (-a * hits / N + s.rho + bin * dp).exp()
        * (1.0 + growth * ce);
    let mut t1 = base * (1.0 + pn * (D / 25.0).ln());
    if s.observed {
        t1 = t1 * (cf * pv).exp() * (dp + pv).cos();
    }

    let t2 = s.scale * s.amplitude + s.trend_bias;

    let valve = bt
        * dp.cos()
        * (N * P / D.sqrt())
        * (1.0 + ch * (D - 25.0) / 25.0)
        * (1.0 + pf * (th + PI).cos() + su * th.sin());
    let sdt = dt.sin();
    let cdt = dt.cos();
    let acoustic = 1.0 + (abl * sdt * sdt) / PHI + (ain * cdt * cdt) / PHI;
    let phase = 1.0 + bin * pv;
    let t3 = valve * acoustic * phase;

    K() * (t1 + t2 + t3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_positive() {
        assert!(PHI > 1.6 && PHI < 1.62);
        assert!(K().is_finite());
        assert!(c_factor().is_finite());
    }

    #[test]
    fn scalar_default_finite() {
        let s = ScalarInput::default();
        let v = compute_scalar(&s);
        assert!(v.is_finite(), "S={v}");
    }

    #[test]
    fn observed_changes_scalar() {
        let mut a = ScalarInput::default();
        a.observed = false;
        let mut b = a.clone();
        b.observed = true;
        assert_ne!(compute_scalar(&a), compute_scalar(&b));
    }
}
