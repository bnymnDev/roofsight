# Security policy

## Supported versions

roofsight is pre-release. Only the `main` branch receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
vulnerability reporting instead:

1. Go to the repository's **Security** tab.
2. Choose **Report a vulnerability** and fill in the advisory form.

Include what you found, how to reproduce it, and which part of the repository it affects
(Python package, CLI, dataset pipeline, export path or the Swift package). You will get a
reply within seven days, and a fix or a decision within thirty. Public credit is given in the
advisory unless you ask otherwise.

## Scope

- The `roofsight` Python package and CLI, including the dataset build, labeling, evaluation
  and export code
- The `RoofGeometry` Swift package
- The GitHub Actions workflows in `.github/workflows/`

Vulnerabilities in third-party dependencies should be reported upstream; a report here is
still welcome if roofsight uses the dependency in a way that makes the problem exploitable.
