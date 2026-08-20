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
                Self::Add(lhs)
            }
            (Self::Add(mut lhs), rhs) => {
                lhs.push(rhs);
                Self::Add(lhs)
            }
            (lhs, Self::Add(mut rhs)) => {
                rhs.insert(0, lhs);
                Self::Add(rhs)
            }
            (lhs, rhs) => Self::Add(vec![lhs, rhs]),
        }
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
                Self::Multiply(lhs)
            }
            (Self::Multiply(mut lhs), rhs) => {
                lhs.push(rhs);
                Self::Multiply(lhs)
            }
            (lhs, Self::Multiply(mut rhs)) => {
                rhs.insert(0, lhs);
                Self::Multiply(rhs)
            }
            (lhs, rhs) => Self::Multiply(vec![lhs, rhs]),
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
}
