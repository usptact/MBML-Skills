using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.VisualBasic.FileIO;

namespace MBMLSkills;

static class DataIO
{
    // ─── Synthetic format ──────────────────────────────────────────────────────

    public static void Write(string dir, SkillsDataset data)
    {
        Directory.CreateDirectory(dir);
        WriteBoolMatrix(Path.Combine(dir, "skills_required.csv"), IndicesToMask(data.SkillsRequired, data.NumSkills));
        WriteBoolMatrix(Path.Combine(dir, "is_correct.csv"), data.IsCorrect);
        if (data.PersonSkills is not null)
            WriteBoolMatrix(Path.Combine(dir, "person_skills.csv"), data.PersonSkills);
    }

    public static SkillsDataset ReadSynthetic(string dir)
    {
        string Require(string name) => RequireFile(Path.Combine(dir, name), "run 'generate' first");
        string personSkillsPath = Path.Combine(dir, "person_skills.csv");
        bool[][]? personSkills = File.Exists(personSkillsPath) ? ReadBoolMatrix(personSkillsPath) : null;
        return SkillsDataset.Create(
            ReadSkillsRequired(Require("skills_required.csv")),
            ReadBoolMatrix(Require("is_correct.csv")),
            personSkills
        );
    }

    // ─── Real data format ──────────────────────────────────────────────────────

    public static SkillsDataset ReadReal(string dir)
    {
        string Require(string name) => RequireFile(Path.Combine(dir, name), $"expected in {dir}");
        var skillsData    = ReadCSV(Require(
            "LearningSkills_Real_Data_Experiments-Original-Inputs-Quiz-SkillsQuestionsMask.csv"));
        var responsesData = ReadCSV(Require(
            "LearningSkills_Real_Data_Experiments-Original-Inputs-RawResponsesAsDictionary.csv"));

        return SkillsDataset.Create(
            ParseSkillsRequired(skillsData),
            BuildIsCorrect(ParseTrueAnswers(responsesData), ParsePersonAnswers(responsesData)),
            ParsePersonSkills(responsesData)
        );
    }

    // ─── Synthetic helpers ─────────────────────────────────────────────────────

    static bool[][] IndicesToMask(int[][] indices, int numSkills)
    {
        return indices.Select(row =>
        {
            bool[] mask = new bool[numSkills];
            foreach (int i in row) mask[i] = true;
            return mask;
        }).ToArray();
    }

    static void WriteBoolMatrix(string path, bool[][] matrix)
    {
        using var writer = new StreamWriter(path);
        foreach (var row in matrix)
            writer.WriteLine(string.Join(",", row.Select(b => b ? 1 : 0)));
    }

    static int[][] ReadSkillsRequired(string path) =>
        File.ReadAllLines(path)
            .Select(line => line.Split(',')
                .Select((v, i) => (v, i))
                .Where(x => x.v == "1")
                .Select(x => x.i)
                .ToArray())
            .ToArray();

    static bool[][] ReadBoolMatrix(string path) =>
        File.ReadAllLines(path)
            .Select(line => line.Split(',').Select(v => v == "1").ToArray())
            .ToArray();

    // ─── Real data helpers ─────────────────────────────────────────────────────

    static string RequireFile(string path, string hint)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Required file not found: {path} — {hint}", path);
        return path;
    }

    static List<string[]> ReadCSV(string path)
    {
        var list = new List<string[]>();
        using var parser = new TextFieldParser(path);
        parser.TextFieldType = FieldType.Delimited;
        parser.SetDelimiters(",");
        while (!parser.EndOfData)
            list.Add(parser.ReadFields()!);
        return list;
    }

    // Counts skill columns in the header row: those starting with "S" after the ID column.
    static int SkillColumnCount(string[] header) =>
        header.Skip(1).TakeWhile(h => h.StartsWith("S")).Count();

    static int[] ParseTrueAnswers(List<string[]> list)
    {
        int qOffset    = 1 + SkillColumnCount(list[0]);
        string[] fields = list[1];
        return Enumerable.Range(qOffset, fields.Length - qOffset).Select(i => int.Parse(fields[i])).ToArray();
    }

    static int[][] ParsePersonAnswers(List<string[]> list)
    {
        int qOffset = 1 + SkillColumnCount(list[0]);
        return list.Skip(2).Select(fields =>
            Enumerable.Range(qOffset, fields.Length - qOffset).Select(i => int.Parse(fields[i])).ToArray()
        ).ToArray();
    }

    static bool[][] ParsePersonSkills(List<string[]> list)
    {
        int numSkills = SkillColumnCount(list[0]);
        return list.Skip(2).Select(fields =>
            Enumerable.Range(1, numSkills).Select(j => fields[j] == "True").ToArray()
        ).ToArray();
    }

    static int[][] ParseSkillsRequired(List<string[]> list) =>
        list.Select(values =>
            values.Select((v, i) => (v, i)).Where(x => x.v == "True").Select(x => x.i).ToArray()
        ).ToArray();

    static bool[][] BuildIsCorrect(int[] trueAnswers, int[][] personAnswers) =>
        personAnswers.Select(answers =>
            Enumerable.Range(0, trueAnswers.Length).Select(q => trueAnswers[q] == answers[q]).ToArray()
        ).ToArray();
}
