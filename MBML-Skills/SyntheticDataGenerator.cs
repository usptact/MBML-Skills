using System;
using System.Linq;

namespace MBMLSkills;

static class SyntheticDataGenerator
{
    public static SkillsDataset Generate(int numSkills, int numQuestions, int numPersons, int? seed = null)
    {
        var rng = seed.HasValue ? new Random(seed.Value) : new Random();
        var skillsRequired = SampleSkillsRequired(numSkills, numQuestions, rng);
        var personSkills   = SamplePersonSkills(numSkills, numPersons, rng);
        var isCorrect      = SampleIsCorrect(numPersons, numQuestions, skillsRequired, personSkills, rng);
        return SkillsDataset.Create(skillsRequired, isCorrect, personSkills);
    }

    static int[][] SampleSkillsRequired(int numSkills, int numQuestions, Random rng)
    {
        return Enumerable.Range(0, numQuestions).Select(q =>
        {
            var list = Enumerable.Range(0, numSkills).Where(s => rng.NextDouble() < 0.5).ToList();
            if (list.Count == 0) list.Add(rng.Next(numSkills));
            return list.ToArray();
        }).ToArray();
    }

    static bool[][] SamplePersonSkills(int numSkills, int numPersons, Random rng) =>
        Enumerable.Range(0, numPersons)
            .Select(p => Enumerable.Range(0, numSkills).Select(s => rng.NextDouble() < 0.5).ToArray())
            .ToArray();

    static bool[][] SampleIsCorrect(int numPersons, int numQuestions,
                                    int[][] skillsRequired, bool[][] personSkills, Random rng) =>
        Enumerable.Range(0, numPersons).Select(p =>
            Enumerable.Range(0, numQuestions).Select(q =>
            {
                bool hasAll = skillsRequired[q].All(s => personSkills[p][s]);
                return rng.NextDouble() < (hasAll ? 0.9 : 0.1);
            }).ToArray()
        ).ToArray();
}
