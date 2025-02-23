# Embedding Explorer
Try it out here: https://gduteaud.github.io/embedding-explorer/

For some reason I often find myself explaining the concept of vector embeddings to non-technical folks, and I have found that good visuals go a long way. What's better than a nice but static illustration, though? An interactive tool, of course! 

## The Cool Technical Details
This project leverages [@xenova](https://github.com/xenova)/[@huggingface](https://github.com/huggingface)'s amazing transformers.js library to run the [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) embedding model entirely client-side, in the user's browser. 

## The Boring Technical Details
This project is set up as a simple static React application with Vite & TypeScript, hosted via GitHub Pages. It has a basic form of CI/CD in place, using GitHub Actions to automatically build and deploy on push. 

## What's Next?
While this first version accomplished what I had in mind when I first got started on this project, I'm now thinking of several ways I can improve it. Longer term, I want to:

- Add a slightly more detailed history/explanation of embeddings
- Provide a few sets of example words/sentences that display interesting relationships
- Allow embedding model selection
- Allow embedding model comparison (to visualize how different models encode inputs differently)
- Implement "sliding window" visualization (animated plot looping over all embedding space dimensions)
- Generate plausible axis labels using an LLM *
- Add a table display showing more (5-10?) dimensions with plausible labels

* I would really like to keep this app running entirely within the browser. Unfortunately my early experiments with this approach using a generic model to generate labels have so far proven unsuccessful, so I am exploring the possibility of fine-tuning a model for this specific purpose.