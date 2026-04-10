---
name: resumeagent-jd-analysis
description: Use when the user wants to analyze a job description, extract required skills, identify interview focus areas, or answer follow-up questions about a JD in the ResumeAgent project.
---

# ResumeAgent JD Analysis

Use this skill when the task is to analyze a job description in the ResumeAgent workflow.

This skill is appropriate when the user:

- provides a JD text block or JD file
- asks for required and preferred skills
- wants interview preparation suggestions based on a JD
- asks follow-up questions about an already analyzed JD

## Inputs

Preferred inputs:

- raw JD text
- uploaded JD file content
- existing `jd_data` from the current session

## Workflow

1. Check whether the JD is already available in structured form.
2. If not, extract the JD into structured fields such as:
   - position
   - responsibilities
   - required skills
   - preferred skills
   - seniority
3. Distinguish between:
   - full JD analysis
   - JD follow-up
4. For full analysis, produce a complete report with:
   - role summary
   - hard requirements
   - soft requirements
   - interview focus areas
   - likely evaluation dimensions
5. For follow-up questions, respond briefly and directly instead of regenerating the full report.

## Output Guidelines

- Prefer structured, decision-oriented answers.
- If the user asks a follow-up question, default to a concise answer.
- Avoid exposing internal planning or model reasoning.
- Use the JD itself as the main source of truth.

## ResumeAgent-specific Notes

- In this project, JD analysis maps naturally to:
  - `jd_analysis`
  - `jd_followup`
- If both JD and resume are already present, consider whether the task is actually a matching task rather than a pure JD task.
- If the user asks "what should I prepare for interview" based on a JD, treat it as a JD follow-up unless resume comparison is explicitly requested.
