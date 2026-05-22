using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Distributions;

namespace MBMLSkills;

static class SyntheticDataGenerator
{
    public static SkillsDataset Generate(int numSkills, int numQuestions, int numPersons)
    {
        var skillsRequired = SampleSkillsRequired(numSkills, numQuestions);
        var personSkills   = SamplePersonSkills(numSkills, numPersons);
        var isCorrect      = SampleIsCorrect(numPersons, numQuestions, skillsRequired, personSkills);
        return SkillsDataset.Create(skillsRequired, isCorrect, personSkills);
    }

    static int[][] SampleSkillsRequired(int numSkills, int numQuestions)
    {
        var coin = new Bernoulli(0.5);
        var rng  = new Random();
        return Enumerable.Range(0, numQuestions).Select(q =>
        {
            var list = Enumerable.Range(0, numSkills).Where(s => coin.Sample()).ToList();
            if (list.Count == 0) list.Add(rng.Next(numSkills));
            return list.ToArray();
        }).ToArray();
    }

    static bool[][] SamplePersonSkills(int numSkills, int numPersons)
    {
        var coin = new Bernoulli(0.5);
        return Enumerable.Range(0, numPersons)
            .Select(p => Enumerable.Range(0, numSkills).Select(s => coin.Sample()).ToArray())
            .ToArray();
    }

    static bool[][] SampleIsCorrect(int numPersons, int numQuestions,
                                    int[][] skillsRequired, bool[][] personSkills)
    {
        var hasSkillsCoin = new Bernoulli(0.9);
        var noSkillsCoin  = new Bernoulli(0.1);
        return Enumerable.Range(0, numPersons).Select(p =>
            Enumerable.Range(0, numQuestions).Select(q =>
            {
                bool hasAll = skillsRequired[q].All(s => personSkills[p][s]);
                return hasAll ? hasSkillsCoin.Sample() : noSkillsCoin.Sample();
            }).ToArray()
        ).ToArray();
    }
}
