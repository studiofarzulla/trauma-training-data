# Training Data and the Maladaptive Mind

**A Computational Framework for Understanding Developmental Trauma**

[![DOI](https://img.shields.io/badge/DOI-10.21203%2Frs.3.rs--8634152%2Fv1-blue.svg)](https://doi.org/10.21203/rs.3.rs-8634152/v1)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-Under_Review-yellow.svg)](https://doi.org/10.21203/rs.3.rs-8634152/v1)

**Working Paper DP-2501** | [Dissensus AI](https://dissensus.ai)

Currently under review at **Humanities & Social Sciences Communications** (Nature).

## Abstract

Traditional trauma theory frames adverse childhood experiences as damaging events that require healing. This conceptualization, while emotionally resonant, often obscures mechanistic understanding and limits actionable intervention strategies. We propose a computational reframing: trauma represents maladaptive learned patterns arising from suboptimal training environments, functionally equivalent to problems observed in machine learning systems trained on poor-quality data. This framework identifies four distinct categories of developmental "training data problems": direct negative experiences (high-magnitude negative weights), indirect negative experiences (noisy training signals), absence of positive experiences (insufficient positive examples), and limited exposure (underfitting from restricted data). We extend this framework to model dissociation as meta-learned protective suppression -- second-order learning where the system learns that cognitive engagement itself predicts overwhelm, producing preemptive processing suppression. This meta-learning model provides mechanistic grounding for the clinical distinction between PTSD (catastrophic single-event learning with localized weight perturbation) and CPTSD (chronic adversarial training with systematic weight landscape distortion), generating testable predictions for differential diagnosis and treatment response. We demonstrate that extreme penalties produce overcorrection and weight cascades in both artificial and biological neural networks, and argue that nuclear family structures constitute limited training datasets prone to overfitting. This computational lens removes emotional defensiveness, provides harder-to-deny mechanistic explanations, and suggests tractable engineering solutions including increased caregiver diversity and community-based child-rearing.

## Key Findings

| Finding | Result |
|---------|--------|
| Extreme penalty gradient amplification | 1,247x at extreme penalties |
| Nuclear vs community significance | p=0.0012, d=3.08 (survives Bonferroni correction) |
| Noisy signals confidence collapse | At 60% noise threshold |
| Limited dataset generalization gap | Scales as 1/sqrt(num_caregivers) |
| Experience replay retention | 93% retention with 71% new learning |

## Computational Models

Four PyTorch implementations validating the framework:

1. **Extreme Penalty** -- Single traumatic event with 1000x penalty produces gradient cascade and overcorrection
2. **Noisy Signals** -- Inconsistent feedback: prediction variance scales as sqrt(p_noise)
3. **Limited Dataset** -- Train on 2-10 caregivers, test on 50: generalization gap ~ 1/sqrt(N)
4. **Catastrophic Forgetting** -- Experience replay balances old retention (93%) with new learning (71%)

## Repository Structure

```
trauma-training-data/
├── paper/                      # LaTeX source and compiled PDF
│   ├── trauma-arxiv.tex        # Preprint version (canonical)
│   └── figures/                # Paper figures
├── trauma-models/              # Computational models (PyTorch)
│   ├── trauma_models/          # Python package
│   │   ├── core/               # Shared base classes
│   │   ├── extreme_penalty/    # Model 1
│   │   ├── noisy_signals/      # Model 2
│   │   ├── limited_dataset/    # Model 3
│   │   └── catastrophic_forgetting/  # Model 4
│   ├── outputs/                # Generated results and figures
│   ├── requirements.txt        # Python dependencies
│   └── run_all_experiments.sh  # Convenience runner
├── CITATION.cff                # Citation metadata
└── LICENSE                     # CC BY 4.0
```

## Replication

```bash
cd trauma-models/
pip install -r requirements.txt

# Run all experiments
./run_all_experiments.sh

# Or individually
python -m trauma_models.extreme_penalty.experiment
python -m trauma_models.noisy_signals.experiment
python -m trauma_models.limited_dataset.experiment
python -m trauma_models.catastrophic_forgetting.experiment
```

Requirements: Python 3.10+, PyTorch 2.0+

## Keywords

Trauma, Machine Learning, Computational Cognitive Science, Philosophy of Mind, Developmental Psychology

## Citation

```bibtex
@article{farzulla2025trauma,
  title={Training Data and the Maladaptive Mind: How Machine Learning Illuminates Developmental Psychopathology},
  author={Farzulla, Murad},
  year={2025},
  doi={10.21203/rs.3.rs-8634152/v1},
  url={https://doi.org/10.21203/rs.3.rs-8634152/v1},
  note={Under Review at Humanities \& Social Sciences Communications (Nature)}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
Computational models: [MIT License](trauma-models/LICENSE)
