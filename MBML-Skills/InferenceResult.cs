using Microsoft.ML.Probabilistic.Distributions;

namespace MBMLSkills;

record InferenceResult(
    Bernoulli[][] SkillMarginals,    // [person][skill]
    Beta[]        ProbGuessMarginals // [question]
);
