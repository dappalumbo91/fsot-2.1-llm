use fsot_core::{c_factor, compute_scalar, ScalarInput, E, G_CAT, GAMMA, PHI, PI, K};

fn main() {
    println!("=== FSOT 2.1 Rust scalar core ===");
    println!("  pi    = {PI}");
    println!("  e     = {E}");
    println!("  phi   = {PHI}");
    println!("  gamma = {GAMMA}");
    println!("  G_cat = {G_CAT}");
    println!("  K     = {}", K());
    println!("  C_f   = {}", c_factor());

    let mut s0 = ScalarInput::default();
    s0.D_eff = 25.0;
    s0.observed = false;
    let mut s1 = s0.clone();
    s1.observed = true;

    println!(
        "  S(D_eff=25, observed=false) = {}",
        compute_scalar(&s0)
    );
    println!(
        "  S(D_eff=25, observed=true)  = {}",
        compute_scalar(&s1)
    );
    println!("FSOT fluid medium online (Rust).");
}
