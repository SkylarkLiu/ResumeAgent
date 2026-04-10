---
name: resumeagent-resume-analysis
description: Use when the user wants to extract, analyze, summarize, or improve a resume in the ResumeAgent project, including resume follow-up questions after an initial analysis.
---

# ResumeAgent Resume Analysis

Use this skill when the task is to analyze a resume in the ResumeAgent workflow.

This skill is appropriate when the user:

- uploads or pastes a resume
- asks for resume strengths and weaknesses
- asks how to improve wording, highlights, or structure
- asks follow-up questions about an already analyzed resume

## Inputs

Preferred inputs:

- raw resume text
- uploaded resume file content
- existing `resume_data` from the current session

## Workflow

1. Check whether structured resume data already exists.
2. If it exists, reuse it instead of re-extracting the resume.
3. If not, extract resume structure, including:
   - summary
   - skills
   - experience
   - projects
   - education
   - target position if available
4. Distinguish between:
   - full resume analysis
   - resume follow-up
5. For full analysis, produce a complete report with:
   - overall assessment
   - strengths
   - gaps
   - wording or structure suggestions
6. For follow-up questions, answer in brief and avoid regenerating a long report unless explicitly requested.

## Output Guidelines

- Default to concise, practical, improvement-oriented advice.
- Highlight the most important changes first.
- Prefer 3-5 high-value suggestions over a long generic checklist.
- Avoid exposing internal chain-of-thought or routing details.

## ResumeAgent-specific Notes

- In this project, resume analysis maps naturally to:
  - `resume_analysis`
  - `resume_followup`
- If a JD is already present in session state and the user is clearly asking about match or fit, switch to the matching workflow instead of a resume-only answer.
- If the user only wants a follow-up answer, use the already extracted `resume_data` whenever possible.
