# Architecture

## Data flow

```text
SPICE text
   |
   v
spice-core lexer/parser  ->  Vec<Statement>
   |
   v
MnaBuilder               ->  symbolic A, K, B, U and ordered X
   |                                      |
   |                                      +-> StringMnaSystem -> PyO3/WASM/PDF
   v
parameter evaluation
   |
   v
NumericMnaSystem         ->  Schur reduction -> NumericStateSpace
```

The parser remains the authority for dialect syntax. `elspice-mna` only
interprets linear element instances and therefore does not duplicate netlist
lexing or statement parsing.

## Unknown ordering

All non-ground node voltages are allocated in first-seen order. Branch currents
for `V`, `L`, `E`, and `H` elements follow, also in first-seen order. Branches
are allocated in a separate pass before any stamp is applied, so an `F` or `H`
element may refer to its controlling voltage source regardless of netlist line
order.

Ground aliases `0` and `gnd` are excluded from the unknown vector.

## Stamp convention

Independent current sources use the SPICE direction from their first node to
their second node. Dependent current sources use the corresponding standard
SPICE sign convention. This may differ from older ElSpice examples that
described the controlled current as entering the first output node.

## State-space reduction

Given

```text
[A_ss A_sa] [x_s] + [K_ss 0] [dot(x_s)] = [B_s] u
[A_as A_aa] [x_a]   [  0   0] [dot(x_a)]   [B_a]
```

the algebraic variables are eliminated using `A_aa` solves:

```text
A_red = A_ss - A_sa A_aa^-1 A_as
B_red = B_s  - A_sa A_aa^-1 B_a

dot(x_s) = -K_ss^-1 A_red x_s + K_ss^-1 B_red u
```

The numeric reducer requires `A_aa` and `K_ss` to be nonsingular. Circuits with
dependent storage coordinates (for example, some floating capacitor networks)
remain valid descriptor systems but need a rank-revealing coordinate transform
before an explicit minimal state-space can be formed.

## Converter phases

Switch overrides replace a named element with `Ron` or `Roff` between its first
two terminals. Since both phases retain the same nodes and branch variables,
their descriptor matrices can be averaged entry by entry.

At a steady operating point, the two-phase duty perturbation is

```text
q_d = (B_on - B_off) U - (A_on - A_off) X
```

and enters the descriptor equation as an additional input column multiplying
the small duty perturbation `d`.

