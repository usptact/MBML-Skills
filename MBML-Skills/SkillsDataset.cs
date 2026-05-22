namespace MBMLSkills;

record SkillsDataset(
    int[][]  SkillsRequired,  // [question][skill_index]
    bool[][] IsCorrect,       // [person][question]
    bool[][] PersonSkills     // [person][skill] — ground truth for display only
)
{
    public int NumPersons   => IsCorrect.Length;
    public int NumQuestions => SkillsRequired.Length;
    public int NumSkills    => PersonSkills[0].Length;
}
