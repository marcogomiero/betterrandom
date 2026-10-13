# Security policy

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private
reporting instead: open the **Security** tab of this repository and choose
**Report a vulnerability**. You will get an answer as soon as possible.

Supported versions: only the latest release receives fixes.

## Threat model

What betterRandom aims to guarantee, assuming the operating system CSPRNG
(`os.urandom`) works as documented:

- Results are unpredictable from past outputs and cannot be reproduced.
- Results are exactly uniform over the requested range. There is no modulo
  bias. This is covered by exhaustive tests on small word sizes.
- The generator is stateless: no buffered output sits in memory, nothing is
  shared between threads, and nothing is duplicated by `fork()`.

What it does not and cannot guarantee:

- The quality of the operating system generator. If it is broken or
  compromised, so is betterRandom. Paranoid mode mixes in extra sources as a
  hedge, but those sources (clocks, process ids, counters) carry little
  entropy, so it is not a substitute for a healthy OS generator.
- Protection against an attacker who can read the memory of your process, or
  who can observe timing and other side channels in the Python interpreter.
- Anything about the continuous distributions of `rng` (`gauss`, `uniform`,
  `betavariate` and the others). They are the standard library algorithms on
  top of 53 bit uniform floats. Only the integer functions are covered by the
  exactness claims.
- Suitability for generating keys, tokens or passwords. For those use the
  standard library `secrets` module.
- Any form of certification. The project has had no external review or audit.

## Scope

Reports about the correctness of the range reduction, the sampling functions
(`better_sample`, `better_choices`, `better_shuffle`), the behavior of
`BetterSystemRandom` on a specific Python version, or the handling of unusual
inputs are very welcome and are in scope.