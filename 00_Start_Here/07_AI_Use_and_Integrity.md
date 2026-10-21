# 07. AI use and academic integrity (please read this before you present)

## What the course policy says
From the *Course Policy*, section "GenAI Usage" (Projects):
- **Allowed:** use AI to "assist in the planning, design, and outlining of project ideas" and to "conduct research, analyze data, and find relevant information". "GenAI is permitted to be used only for the deployment of model or learning new concepts beyond classroom. Students need to cite the references used."
- **Not allowed:** "Develop entire project; the core implementation and critical thinking aspects of the project must be the students' own work." And: "Use AI in ways that bypass the learning process and objectives of the course."
- General rule for assignments: do not use AI "for creating entire code or solutions using AI without understanding and personal contribution" and do not copy "AI-generated content and pass it off as their own work".
- AI tools are **not allowed during exams and quizzes**.

## The honest situation
This project's code, lessons, tests and diagrams were developed in a long working session between **Advik and an AI coding assistant (Claude)**. Advik directed the work, made decisions and reviewed results; the assistant wrote most of the code and documents. The folder you are reading, including the viva answers, was also generated with AI help.

That does not make the project worthless, but it does mean two things you must handle honestly:
1. **The policy expects the core implementation and critical thinking to be yours.** Be ready to show it: you must understand every line of your part, and ideally have done real work in it yourself (re-run, modify, extend, write in your own words).
2. **You must declare the AI help.** Hiding it, if discovered, is a much bigger problem than the help itself.

## What to do (practical, in order)
1. **Tell the coordinator now**, in a short email, how AI was used and ask whether the project is acceptable as it stands and what they expect from you personally. Do this before the presentation, not after a question in the viva.
2. **Write the declaration** in the report (draft below, edit so it is true for you).
3. **Do real work on your own part**, so that the declaration is honest and the viva is easy:
   - re-run your notebooks from the first cell and write down what each output means in your own words;
   - change one thing (a threshold, a layer, a feature, a split) and record what happened and why;
   - add one small experiment of your own (the optional extras in `04_Course_Mapping.md` are good candidates);
   - write your report section yourself; use `02_Knowledge_Pack.md` to learn, not to copy.
4. **Cite sources.** Papers are in `04_Course_Mapping.md`; also cite software (PyTorch, XGBoost, scikit-learn, astropy, SEP) and data (ZTF via IRSA, with the acknowledgement text from IRSA's release notes, https://irsa.ipac.caltech.edu/data/ZTF/docs/releases/ztf_release_notes_latest).
5. **In the viva, do not bluff.** "That part is Paras's; from my understanding it works like this..." is fine.

## Draft declaration (edit to match what is true for you)
> **Use of generative AI.** This project was developed with the assistance of an AI coding assistant (Claude, by Anthropic), used under the direction of Advik Divekar. The assistant helped to plan the project, write and test the source code, write the lesson documents and diagrams, and prepare study material. Each team member then studied, re-ran and [describe what you personally did: e.g. modified and evaluated] the part they present: Advik (system, evaluation and deep search), Aaryan (data and preprocessing), Paras (labels and the neural network), Satvik (linking, tree models and metrics). We understand and can explain the methods and results, and all references used are cited. [Add anything else that is true: experiments you added, data you inspected yourself, errors you found.]

## A note about the project itself
Because the policy is about *learning*, the most valuable thing you can do with this folder is use it the way a student uses a worked example: read, run, break it on purpose, explain it out loud, and then extend it.
