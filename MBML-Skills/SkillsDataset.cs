using System;
using System.Linq;

namespace MBMLSkills;

record SkillsDataset(
    int[][]  SkillsRequired,  // [question][skill_index]
    bool[][] IsCorrect,       // [person][question]
    bool[][] PersonSkills     // [person][skill] — ground truth for display only
)
{
    public int NumPersons   => IsCorrect.Length;
    public int NumQuestions => SkillsRequired.Length;
    public int NumSkills    => PersonSkills.Length > 0 ? PersonSkills[0].Length : 0;

    public static SkillsDataset Create(int[][] skillsRequired, bool[][] isCorrect, bool[][] personSkills)
    {
        if (personSkills.Length != isCorrect.Length)
            throw new InvalidOperationException(
                $"PersonSkills has {personSkills.Length} persons but IsCorrect has {isCorrect.Length}.");

        int badRow = Array.FindIndex(isCorrect, row => row.Length != skillsRequired.Length);
        if (badRow >= 0)
            throw new InvalidOperationException(
                $"IsCorrect[{badRow}] has {isCorrect[badRow].Length} entries " +
                $"but SkillsRequired has {skillsRequired.Length} questions.");

        int numSkills     = personSkills.Length > 0 ? personSkills[0].Length : 0;
        int maxSkillIndex = skillsRequired.SelectMany(q => q).DefaultIfEmpty(-1).Max();
        if (maxSkillIndex >= numSkills)
            throw new InvalidOperationException(
                $"SkillsRequired references skill index {maxSkillIndex} " +
                $"but PersonSkills has only {numSkills} columns.");

        return new SkillsDataset(skillsRequired, isCorrect, personSkills);
    }
}
