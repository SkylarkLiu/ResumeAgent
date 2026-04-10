---
name: resumeagent-match-analysis
description: Use when the user wants to compare a resume against a JD, identify gaps, prioritize improvements, or answer follow-up match questions in the ResumeAgent project.
---

# ResumeAgent Match Analysis

Use this skill when the task is to compare a resume with a job description.

This skill is appropriate when the user:

- asks whether a resume matches a JD
- asks what is missing relative to a target job
- asks what to improve first for a specific job
- has already analyzed both JD and resume and wants targeted follow-up advice

## Inputs

Preferred inputs:

- structured `jd_data`
- structured `resume_data`
- follow-up question that expresses match, gap, fit, or priority

## Workflow

1. Check whether both JD and resume are available.
2. If one is missing, identify the missing side and request or infer it from session state.
3. If both are available, compare:
   - must-have skills
   - preferred skills
   - projects and evidence
   - experience depth
   - role alignment
4. Distinguish between:
   - full match analysis
   - match follow-up
5. For full analysis, produce:
   - overall match judgment
   - major strengths
   - major gaps
   - top priorities to improve
6. For follow-up questions, give a short, direct answer focused on:
   - what is missing
   - what matters most
   - what to improve first

## Output Guidelines

- Lead with the conclusion.
- Prioritize the top 3 gaps or improvement directions.
- Keep follow-up answers brief and actionable.
- Do not regenerate a full report unless the user explicitly asks for one.

## ResumeAgent-specific Notes

- In this project, match analysis maps naturally to:
  - `match_followup`
  - `resume_analysis` with JD context
- Reuse `resume_data`, `jd_data`, `working_context`, and `context_sources` whenever available.
- If the user asks "what should I补/缺什么/优先改什么" after both sides are available, treat it as a match follow-up rather than a fresh full analysis.
