use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A small symbolic expression tree used in MNA stamps.
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// A numeric constant.
    Constant(f64),
    /// A named component, parameter, source value, or duty ratio.
    Symbol(String),
    /// A sum.
    Add(Vec<Expression>),
    /// A product.
    Multiply(Vec<Expression>),
    /// A reciprocal.
    Reciprocal(Box<Expression>),
    /// A square root.
    Sqrt(Box<Expression>),
}

impl Expression {
    /// Additive identity.
    pub const fn zero() -> Self {
        Self::Constant(0.0)
    }

    /// Multiplicative identity.
    pub const fn one() -> Self {
        Self::Constant(1.0)
    }

    /// Creates a symbol.
    pub fn symbol(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }

    /// Creates `1 / self`.
    pub fn reciprocal(self) -> Self {
        match self {
            Self::Constant(value) => Self::Constant(1.0 / value),
            // 1/(1/x) = x. Without this, repeated symbolic elimination (each
            // pivot normalizes a row by its own reciprocal) builds deeply
            // nested reciprocal chains that never combine with the plain
            // occurrences of the same sub-expression elsewhere, which in turn
            // defeats the like-term cancellation in `Add`/`Mul` below.
            Self::Reciprocal(inner) => *inner,
            other => Self::Reciprocal(Box::new(other)),
        }
    }

    /// Creates `sqrt(self)`.
    pub fn sqrt(self) -> Self {
        match self {
            Self::Constant(value) if value >= 0.0 => Self::Constant(value.sqrt()),
            other => Self::Sqrt(Box::new(other)),
        }
    }

    /// Returns true for an exact zero constant.
    pub fn is_zero(&self) -> bool {
        matches!(self, Self::Constant(value) if *value == 0.0)
    }

    /// Evaluates the expression using symbol values.
    pub fn evaluate(&self, values: &BTreeMap<String, f64>) -> Result<f64, EvaluationError> {
        match self {
            Self::Constant(value) => Ok(*value),
            Self::Symbol(name) => values
                .get(name)
                .or_else(|| {
                    values
                        .iter()
                        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
                        .map(|(_, value)| value)
                })
                .copied()
                .ok_or_else(|| EvaluationError::MissingSymbol(name.clone())),
            Self::Add(terms) => terms
                .iter()
                .try_fold(0.0, |sum, term| Ok(sum + term.evaluate(values)?)),
            Self::Multiply(factors) => factors.iter().try_fold(1.0, |product, factor| {
                Ok(product * factor.evaluate(values)?)
            }),
            Self::Reciprocal(value) => {
                let denominator = value.evaluate(values)?;
                if denominator == 0.0 {
                    Err(EvaluationError::DivisionByZero)
                } else {
                    Ok(1.0 / denominator)
                }
            }
            Self::Sqrt(value) => {
                let radicand = value.evaluate(values)?;
                if radicand < 0.0 {
                    Err(EvaluationError::NegativeSquareRoot(radicand))
                } else {
                    Ok(radicand.sqrt())
                }
            }
        }
    }

    /// Collects all symbol names used by the expression.
    pub fn symbols(&self) -> BTreeSet<String> {
        let mut result = BTreeSet::new();
        self.collect_symbols(&mut result);
        result
    }

    fn collect_symbols(&self, result: &mut BTreeSet<String>) {
        match self {
            Self::Symbol(name) => {
                result.insert(name.clone());
            }
            Self::Add(values) | Self::Multiply(values) => {
                for value in values {
                    value.collect_symbols(result);
                }
            }
            Self::Reciprocal(value) | Self::Sqrt(value) => value.collect_symbols(result),
            Self::Constant(_) => {}
        }
    }

    pub(crate) fn parse_scalar(raw: &str) -> Result<Self, String> {
        let trimmed = raw
            .trim()
            .trim_matches(|c| matches!(c, '{' | '}' | '\'' | '"'));
        if trimmed.is_empty() {
            return Err("empty scalar value".into());
        }
        match parse_spice_number(trimmed) {
            Some(value) => Ok(Self::Constant(value)),
            None if is_simple_symbol(trimmed) => Ok(Self::Symbol(trimmed.to_string())),
            None => Err(format!(
                "unsupported scalar expression '{raw}'; use a number or symbol"
            )),
        }
    }
}

impl Add for Expression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        match (self, rhs) {
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a + b),
            (Self::Add(mut lhs), Self::Add(rhs)) => {
                lhs.extend(rhs);
                combine_terms(lhs)
            }
            (Self::Add(mut lhs), rhs) => {
                lhs.push(rhs);
                combine_terms(lhs)
            }
            (lhs, Self::Add(mut rhs)) => {
                rhs.insert(0, lhs);
                combine_terms(rhs)
            }
            (lhs, rhs) => combine_terms(vec![lhs, rhs]),
        }
    }
}

/// Splits a term into `(coefficient, base)` so that e.g. `x` and `-1 * x`
/// (built by [`Neg`]) are recognized as the same base with opposite sign.
fn coefficient_and_base(expr: &Expression) -> (f64, Expression) {
    if let Expression::Multiply(factors) = expr {
        if let Some(Expression::Constant(value)) = factors.first() {
            let rest = &factors[1..];
            let base = if rest.len() == 1 {
                rest[0].clone()
            } else {
                Expression::Multiply(rest.to_vec())
            };
            return (*value, base);
        }
    }
    (1.0, expr.clone())
}

/// Combines a flat list of addends, folding constants together and summing
/// the coefficients of any terms that share the same base (e.g. `x + (-x)`
/// cancels to `0`, `2*x + 3*x` combines to `5*x`). This is what lets
/// structural (exact-zero) singularity detection actually see a
/// mathematically-zero combination that Gaussian elimination produced from
/// otherwise-distinct-looking sub-expressions.
fn combine_terms(terms: Vec<Expression>) -> Expression {
    let mut constant_sum = 0.0;
    let mut combined: Vec<(f64, Expression)> = Vec::new();
    for term in terms {
        if let Expression::Constant(value) = term {
            constant_sum += value;
            continue;
        }
        let (coefficient, base) = coefficient_and_base(&term);
        if let Some(existing) = combined.iter_mut().find(|(_, other)| *other == base) {
            existing.0 += coefficient;
        } else {
            combined.push((coefficient, base));
        }
    }

    let mut result_terms = Vec::new();
    if constant_sum != 0.0 {
        result_terms.push(Expression::Constant(constant_sum));
    }
    for (coefficient, base) in combined {
        if coefficient == 0.0 {
            continue;
        } else if coefficient == 1.0 {
            result_terms.push(base);
        } else if coefficient == -1.0 {
            result_terms.push(Expression::Multiply(vec![Expression::Constant(-1.0), base]));
        } else {
            result_terms.push(Expression::Multiply(vec![
                Expression::Constant(coefficient),
                base,
            ]));
        }
    }

    match result_terms.len() {
        0 => Expression::zero(),
        1 => result_terms.into_iter().next().expect("length checked"),
        _ => Expression::Add(result_terms),
    }
}

impl Neg for Expression {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            Self::Constant(value) => Self::Constant(-value),
            Self::Multiply(mut factors) if matches!(factors.first(), Some(Self::Constant(_))) => {
                if let Self::Constant(value) = &mut factors[0] {
                    *value = -*value;
                }
                Self::Multiply(factors)
            }
            other => Self::Multiply(vec![Self::Constant(-1.0), other]),
        }
    }
}

impl Sub for Expression {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for Expression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        if self == Self::one() {
            return rhs;
        }
        if rhs == Self::one() {
            return self;
        }
        match (self, rhs) {
            (Self::Constant(a), Self::Constant(b)) => Self::Constant(a * b),
            (Self::Multiply(mut lhs), Self::Multiply(rhs)) => {
                lhs.extend(rhs);
                combine_factors(lhs)
            }
            (Self::Multiply(mut lhs), rhs) => {
                lhs.push(rhs);
                combine_factors(lhs)
            }
            (lhs, Self::Multiply(mut rhs)) => {
                rhs.insert(0, lhs);
                combine_factors(rhs)
            }
            (lhs, rhs) => combine_factors(vec![lhs, rhs]),
        }
    }
}

/// Combines a flat list of factors: folds constants together and cancels any
/// base expression that appears with a net multiplicative power of zero
/// (e.g. `x * (1/x)`, or `x * (1/(-1 * x))` which nets to `-1`, since the `x`
/// inside the reciprocal's product also cancels). This is what breaks
/// otherwise-unrecognizable "divide by a pivot, then multiply back related
/// factors" patterns that Gaussian elimination produces, letting them
/// collapse into a small literal result instead of staying as an opaque
/// product that `combine_terms` above can't see through.
fn combine_factors(factors: Vec<Expression>) -> Expression {
    let mut constant_product = 1.0;
    let mut atoms: Vec<(Expression, i32)> = Vec::new();
    for factor in factors {
        absorb_factor(factor, 1, &mut constant_product, &mut atoms);
    }

    if constant_product == 0.0 {
        return Expression::zero();
    }

    let mut result_factors = Vec::new();
    if constant_product != 1.0 {
        result_factors.push(Expression::Constant(constant_product));
    }

    let mut denominator_factors = Vec::new();
    for (atom, power) in atoms {
        if power > 0 {
            for _ in 0..power {
                result_factors.push(atom.clone());
            }
        } else {
            for _ in 0..power.abs() {
                denominator_factors.push(atom.clone());
            }
        }
    }
    // Group every negative-power atom into a single reciprocal of their
    // product (e.g. `1/R * 1/C` -> `1 / (R * C)`) rather than one Reciprocal
    // node per atom.
    match denominator_factors.len() {
        0 => {}
        1 => result_factors.push(Expression::Reciprocal(Box::new(
            denominator_factors
                .into_iter()
                .next()
                .expect("length checked"),
        ))),
        _ => result_factors.push(Expression::Reciprocal(Box::new(Expression::Multiply(
            denominator_factors,
        )))),
    }

    match result_factors.len() {
        0 => Expression::one(),
        1 => result_factors.into_iter().next().expect("length checked"),
        _ => Expression::Multiply(result_factors),
    }
}

/// Recursively decomposes `factor` (contributing `power`, `+1` or `-1`) into
/// a running constant product and a list of `(base, net power)` atoms.
/// Descends into `Reciprocal` (flipping the sign of `power`) and `Multiply`
/// (distributing `power` over every inner factor), so a base expression that
/// occurs both directly and inside an unrelated-looking reciprocal product
/// still nets out and cancels.
fn absorb_factor(
    factor: Expression,
    power: i32,
    constant_product: &mut f64,
    atoms: &mut Vec<(Expression, i32)>,
) {
    match factor {
        Expression::Constant(value) => {
            if power > 0 {
                for _ in 0..power {
                    *constant_product *= value;
                }
            } else {
                for _ in 0..power.abs() {
                    *constant_product /= value;
                }
            }
        }
        Expression::Reciprocal(inner) => {
            absorb_factor(*inner, -power, constant_product, atoms);
        }
        Expression::Multiply(inner_factors) => {
            for inner in inner_factors {
                absorb_factor(inner, power, constant_product, atoms);
            }
        }
        other => {
            if let Some(existing) = atoms.iter_mut().find(|(base, _)| *base == other) {
                existing.1 += power;
            } else {
                atoms.push((other, power));
            }
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_with_precedence(f, 0)
    }
}

impl Expression {
    fn fmt_with_precedence(&self, f: &mut fmt::Formatter<'_>, parent: u8) -> fmt::Result {
        match self {
            Self::Constant(value) => {
                if value.fract() == 0.0 {
                    write!(f, "{value:.0}")
                } else {
                    write!(f, "{value}")
                }
            }
            Self::Symbol(name) => f.write_str(name),
            Self::Add(terms) => {
                let wrap = parent > 1;
                if wrap {
                    f.write_str("(")?;
                }
                for (index, term) in terms.iter().enumerate() {
                    if index > 0 {
                        f.write_str(" + ")?;
                    }
                    term.fmt_with_precedence(f, 1)?;
                }
                if wrap {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::Multiply(factors) => {
                let wrap = parent > 2;
                if wrap {
                    f.write_str("(")?;
                }
                for (index, factor) in factors.iter().enumerate() {
                    if index > 0 {
                        f.write_str(" * ")?;
                    }
                    factor.fmt_with_precedence(f, 2)?;
                }
                if wrap {
                    f.write_str(")")?;
                }
                Ok(())
            }
            Self::Reciprocal(value) => {
                f.write_str("1 / ")?;
                value.fmt_with_precedence(f, 3)
            }
            Self::Sqrt(value) => write!(f, "sqrt({value})"),
        }
    }
}

/// Error returned while numerically evaluating an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// No value was supplied for a symbol.
    MissingSymbol(String),
    /// A reciprocal evaluated to division by zero.
    DivisionByZero,
    /// A square root received a negative value.
    NegativeSquareRoot(f64),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSymbol(name) => write!(f, "missing value for symbol '{name}'"),
            Self::DivisionByZero => f.write_str("division by zero"),
            Self::NegativeSquareRoot(value) => {
                write!(f, "cannot take the real square root of {value}")
            }
        }
    }
}

impl std::error::Error for EvaluationError {}

fn is_simple_symbol(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.'))
}

pub(crate) fn parse_spice_number(raw: &str) -> Option<f64> {
    if let Ok(value) = raw.parse::<f64>() {
        return Some(value);
    }
    let split = raw
        .char_indices()
        .find(|(_, character)| character.is_ascii_alphabetic())
        .map(|(index, _)| index)?;
    let base = raw[..split].parse::<f64>().ok()?;
    let suffix = raw[split..].to_ascii_lowercase();
    let factor = if suffix.starts_with("meg") {
        1e6
    } else {
        match suffix.chars().next()? {
            't' => 1e12,
            'g' => 1e9,
            'k' => 1e3,
            'm' => 1e-3,
            'u' => 1e-6,
            'n' => 1e-9,
            'p' => 1e-12,
            'f' => 1e-15,
            _ => return None,
        }
    };
    Some(base * factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_spice_suffixes() {
        assert_eq!(
            Expression::parse_scalar("1meg"),
            Ok(Expression::Constant(1e6))
        );
        assert_eq!(
            Expression::parse_scalar("2k"),
            Ok(Expression::Constant(2e3))
        );
        assert_eq!(
            Expression::parse_scalar("47u"),
            Ok(Expression::Constant(47e-6))
        );
        assert_eq!(
            Expression::parse_scalar("10M"),
            Ok(Expression::Constant(10e-3))
        );
    }

    #[test]
    fn add_cancels_a_term_with_its_negation() {
        let x = Expression::symbol("x");
        let sum = x.clone() + (-x);
        assert_eq!(sum, Expression::zero());
    }

    #[test]
    fn add_combines_matching_bases_into_a_single_scaled_term() {
        let x = Expression::symbol("x");
        let sum = x.clone() + x.clone() + x;
        assert_eq!(
            sum,
            Expression::Multiply(vec![Expression::Constant(3.0), Expression::symbol("x")])
        );
    }

    #[test]
    fn mul_cancels_a_factor_with_its_reciprocal() {
        let x = Expression::symbol("x");
        let product = x.clone() * x.reciprocal();
        assert_eq!(product, Expression::one());
    }

    #[test]
    fn reciprocal_of_reciprocal_collapses() {
        let x = Expression::symbol("x");
        assert_eq!(x.clone().reciprocal().reciprocal(), x);
    }

    #[test]
    fn mul_cancels_a_factor_against_a_reciprocal_of_a_compound_product() {
        // x * (1/(-1 * x)) = 1 / -1 = -1, since the `x` inside the
        // reciprocal's own product also nets against the plain `x` factor.
        // This is the exact shape Gaussian elimination produces when it
        // divides by a pivot and later multiplies back a related factor.
        let x = Expression::symbol("x");
        let denominator = Expression::Multiply(vec![Expression::Constant(-1.0), x.clone()]);
        let product = x * denominator.reciprocal();
        assert_eq!(product, Expression::Constant(-1.0));
    }

    #[test]
    fn mul_groups_separate_reciprocals_into_one() {
        // 1/R * 1/C -> 1 / (R * C), not two separate Reciprocal factors.
        let r = Expression::symbol("R");
        let c = Expression::symbol("C");
        let product = r.clone().reciprocal() * c.clone().reciprocal();
        assert_eq!(
            product,
            Expression::Reciprocal(Box::new(Expression::Multiply(vec![r, c])))
        );
        assert_eq!(product.to_string(), "1 / (R * C)");
    }
}
