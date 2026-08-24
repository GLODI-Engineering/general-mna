use std::collections::BTreeMap;

use general_mna::MnaBuilder;
use general_spice_core::Dialect;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let netlist = "V1 in 0 1\nR1 in out R\nC1 out 0 C";
    let symbolic = MnaBuilder::new(Dialect::Ngspice).build_fragment(netlist)?;

    println!("unknowns: {:?}", symbolic.unknowns);
    println!("A: {:?}", symbolic.to_string_system().a);
    println!("K: {:?}", symbolic.to_string_system().k);

    let values = BTreeMap::from([("R".into(), 1_000.0), ("C".into(), 1e-6)]);
    let state_space = symbolic.evaluate(&values)?.to_state_space(1e-12)?;
    println!("states: {:?}", state_space.states);
    println!("state A: {:?}", state_space.a.as_slice());
    println!("state B: {:?}", state_space.b.as_slice());
    Ok(())
}
