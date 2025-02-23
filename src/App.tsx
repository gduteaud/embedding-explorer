import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import Plot from 'react-plotly.js'
import { PCA } from 'ml-pca'
import { pipeline, env } from '@xenova/transformers'

env.allowLocalModels = false;

function App() {
  const [inputs, setInputs] = useState<string[]>(['', '', ''])
  const [embeddings, setEmbeddings] = useState<number[][]>()
  const [isModelLoading, setIsModelLoading] = useState(true)
  const [isGenerating, setIsGenerating] = useState(false)
  const [progressItems, setProgressItems] = useState<any[]>([])
  const [error, setError] = useState<string | null>(null)
  const [reducedEmbeddings, setReducedEmbeddings] = useState<number[][]>()
  // const [axisLabels, setAxisLabels] = useState({ x: 'Dimension 1', y: 'Dimension 2', z: 'Dimension 3' });
  
  const worker = useRef<Worker | null>(null);
  const generatorRef = useRef<any>(null);  // Store the pipeline instance

  // Comment out the generateAxisLabels function
  /*const generateAxisLabels = useCallback(async (words: string[], embeddings: number[][]) => {
    console.log('generateAxisLabels called with:', { words, embeddings });

    try {
      if (!generatorRef.current) {
        console.log('Creating text generation pipeline...');
        try {
          generatorRef.current = await pipeline(
            "text-generation",
            "Xenova/distilgpt2",
            {
              cache_dir: './models',
              progress_callback: (progress) => {
                console.log('Model loading progress:', progress);
              }
            }
          );
          console.log('Pipeline created successfully');
        } catch (pipelineError) {
          console.error("Failed to initialize text generation pipeline:", pipelineError);
          return {
            x: "Dimension 1",
            y: "Dimension 2",
            z: "Dimension 3"
          };
        }
      }
      
      const prompt =
      `Given these words with their coordinates (normalized between 0-1):
      ${words.map((word, i) => `"${word}": [${embeddings[i].join(", ")}]`).join("\n")}
      Based on these word positions, suggest three short, meaningful labels for:
      - X-axis (Dimension 1)
      - Y-axis (Dimension 2)
      - Z-axis (Dimension 3) (if applicable)
      Format: "x: label | y: label | z: label"`;

      console.log('About to generate axis labels with prompt:', prompt);
      
      const output = await generatorRef.current(prompt, {
        max_length: 500,
        do_sample: true,
        temperature: 0.0,
      });

      console.log('Generated axis labels output:', output);
      
      const response = output[0].generated_text.trim();
      console.log('Generated axis labels response:', response);
      const labels = response.match(/x: (.*?) \| y: (.*?)( \| z: (.*?))?$/i);
      
      return {
        x: labels?.[1] || "Dimension 1",
        y: labels?.[2] || "Dimension 2",
        z: labels?.[4] || "Dimension 3"
      };
    } catch (error) {
      console.error("Error generating axis labels:", error);
      return {
        x: "Dimension 1",
        y: "Dimension 2",
        z: "Dimension 3"
      };
    }
  }, []);*/

  const processEmbeddings = useCallback(async (embeddings: number[][], originalTexts: string[]) => {
    const inputCount = embeddings.length;
    console.log('Processing embeddings with inputs length:', inputCount);
    const pca = new PCA(embeddings);
    const components = inputCount > 3 ? 3 : 2;
    console.log('Number of PCA components:', components);
    
    const reduced = pca.predict(embeddings, { nComponents: components });
    console.log('Reduced data shape:', reduced.rows, 'x', reduced.columns);
    
    const reducedArray = reduced.data.map((_, i) => {
      const dims = inputCount > 3
        ? [(reduced.get(i, 0) + 1) / 2, (reduced.get(i, 1) + 1) / 2, (reduced.get(i, 2) + 1) / 2]
        : [(reduced.get(i, 0) + 1) / 2, (reduced.get(i, 1) + 1) / 2];
      console.log(`Reduced vector ${i}:`, dims);
      return dims;
    });
    
    console.log('Final reduced embeddings:', reducedArray);

    // Set the reduced embeddings
    setReducedEmbeddings(reducedArray);
    
    // Set default axis labels instead of generating them
    /* setAxisLabels({
      x: "Dimension 1",
      y: "Dimension 2",
      z: "Dimension 3"
    });*/
  }, []); // Remove generateAxisLabels from dependencies

  useEffect(() => {
    console.log('Setting up worker...');
    if (!worker.current) {
      try {
        // Create worker instance
        worker.current = new Worker(new URL('./worker.ts', import.meta.url), {
          type: 'module'
        });
        console.log('Worker created successfully');

        // Set up message handler
        const onMessageReceived = (e: MessageEvent) => {
          console.log('Message received from worker:', e.data);
          switch (e.data.status) {
            case 'loading':
              setProgressItems(prev => {
                const exists = prev.find(item => item.file === e.data.file);
                if (exists) {
                  return prev.map(item => 
                    item.file === e.data.file 
                      ? { ...item, progress: e.data.progress }
                      : item
                  );
                }
                return [...prev, e.data];
              });
              break;

            case 'ready':
              setIsModelLoading(false);
              setProgressItems([]);
              break;

            case 'complete':
              setEmbeddings(e.data.output);
              processEmbeddings(e.data.output, e.data.texts);
              setIsGenerating(false);
              break;

            case 'error':
              setError(e.data.error);
              setIsGenerating(false);
              setIsModelLoading(false);
              break;
          }
        };

        // Add event listener and send initial load message
        worker.current.addEventListener('message', onMessageReceived);
        worker.current.postMessage({ type: 'load' });
        console.log('Load message sent to worker');

        // Cleanup function
        return () => {
          if (worker.current) {
            worker.current.removeEventListener('message', onMessageReceived);
            worker.current.terminate();
            worker.current = null;
          }
        };
      } catch (error) {
        console.error('Worker creation failed:', error);
        setError('Failed to initialize worker: ' + (error as Error).message);
      }
    }
  }, []);

  const generateEmbeddings = async () => {
    if (!worker.current) return;
    
    setIsGenerating(true);
    setEmbeddings(undefined);
    setError(null);
    
    worker.current.postMessage({
      texts: inputs
    });
  };

  useEffect(() => {
    console.log('reducedEmbeddings updated:', reducedEmbeddings);
  }, [reducedEmbeddings]);

  const addInput = () => {
    setInputs([...inputs, '']);
  };

  const removeInput = (index: number) => {
    if (inputs.length > 3) {
      const newInputs = inputs.filter((_, i) => i !== index);
      setInputs(newInputs);
      setReducedEmbeddings(undefined); // Clear plot when removing input
    }
  };

  const handleInputChange = (index: number, value: string) => {
    const newInputs = [...inputs];
    newInputs[index] = value;
    setInputs(newInputs);
    setReducedEmbeddings(undefined); // Clear plot when modifying input
  };

  return (
    <>
      <header>
        <h1>
          Embedding explorer
        </h1>
          <a
            href="https://github.com/gduteaud/embedding-explorer"
            target="_blank"
            rel="noopener noreferrer"
            style={{ marginLeft: '10px', verticalAlign: 'middle' }}
          >
            <img
              src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
              alt="GitHub repository"
              className="github-logo"
            />
          </a>
      </header>
      <p>This is a simple tool to help non-experts visually explore and build intuition for the concept of word/sentence embeddings using the <a href="https://github.com/huggingface/transformers.js">transformers.js</a> library.</p>
            <div className="card">
        <div className="left-container">
          <div className="inputs-container">
            {inputs.map((input, index) => (
              <div key={index} className="input-row">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => handleInputChange(index, e.target.value)}
                  placeholder={`Enter text ${index + 1}`}
                  disabled={isModelLoading || isGenerating}
                />
                {inputs.length > 3 && (
                  <button
                    onClick={() => removeInput(index)}
                    disabled={isModelLoading || isGenerating}
                    className="remove-input"
                  >
                    ✕
                  </button>
                )}
              </div>
            ))}
            <button
              onClick={addInput}
              disabled={
                isModelLoading ||
                isGenerating ||
                inputs.some(input => !input.trim())
              }
              className="add-input"
            >
              + Add input
            </button>
          </div>
        <button 
          onClick={generateEmbeddings} 
          disabled={
            isModelLoading || 
            isGenerating || 
            inputs.length < 3 || 
            inputs.some(input => !input.trim())
          }
        >
          {isModelLoading ? 'Loading model...' : 
           isGenerating ? 'Generating...' : 
           'Generate embeddings'}
        </button>

        {progressItems.length > 0 && (
          <div className="progress-container">
            <p>Loading model files...</p>
            {progressItems.map(item => (
              <div key={item.file}>
                {item.file}: {item.progress?.toFixed(2)}%
              </div>
            ))}
          </div>
        )}

        {error && (
          <div className="error-message">
            Error: {error}
          </div>
        )}
        </div>

        {reducedEmbeddings && (
          <div>
            {inputs.length <= 3 ? (
              <Plot
                data={[
                  {
                    type: 'scatter',
                    x: reducedEmbeddings.map(e => e[0]),
                    y: reducedEmbeddings.map(e => e[1]),
                    text: inputs,
                    mode: 'text+markers',
                    textposition: 'top center',
                    marker: {
                      size: 8,
                      color: 'rgb(100, 108, 255)'
                    }
                  }
                ]}
                layout={{
                  width: 600,
                  height: 500,
                  title: '2D Embedding Visualization',
                  showlegend: false,
                  paper_bgcolor: '#242424',
                  plot_bgcolor: '#242424',
                  xaxis: { 
                    title: { text: 'Dimension 1' },
                    range: [0, 1],
                    gridcolor: '#404040',
                    color: '#fff'
                  },
                  yaxis: { 
                    title: { text: 'Dimension 2' },
                    range: [0, 1],
                    gridcolor: '#404040',
                    color: '#fff'
                  },
                  font: {
                    color: '#fff'
                  }
                }}
              />
            ) : (
              <Plot
                data={[
                  {
                    type: 'scatter3d',
                    x: reducedEmbeddings.map(e => e[0]),
                    y: reducedEmbeddings.map(e => e[1]),
                    z: reducedEmbeddings.map(e => e[2]),
                    text: inputs,
                    mode: 'text+markers',
                    textposition: 'top center',
                    marker: {
                      size: 4,
                      color: 'rgb(100, 108, 255)'
                    }
                  },
                  {
                    type: 'scatter3d',
                    x: reducedEmbeddings.flatMap(e => [0, e[0], null]),
                    y: reducedEmbeddings.flatMap(e => [0, e[1], null]),
                    z: reducedEmbeddings.flatMap(e => [0, e[2], null]),
                    mode: 'lines',
                    line: {
                      color: 'rgb(100, 108, 255)',
                      width: 2
                    },
                    opacity: 0.25,
                    hoverinfo: 'none'
                  }
                ]}
                layout={{
                  width: 600,
                  height: 500,
                  title: '3D Embedding Visualization',
                  showlegend: false,
                  paper_bgcolor: '#242424',
                  scene: {
                    xaxis: { 
                      title: { text: 'Dimension 1' },
                      range: [0, 1],
                      showgrid: true,
                      zeroline: true,
                      gridcolor: '#404040',
                      color: '#fff'
                    },
                    yaxis: { 
                      title: { text: 'Dimension 2' },
                      range: [0, 1],
                      showgrid: true,
                      zeroline: true,
                      gridcolor: '#404040',
                      color: '#fff'
                    },
                    zaxis: { 
                      title: { text: 'Dimension 3' },
                      range: [0, 1],
                      showgrid: true,
                      zeroline: true,
                      gridcolor: '#404040',
                      color: '#fff'
                    },
                    bgcolor: '#242424',
                    camera: {
                      eye: {x: 1.5, y: 1.5, z: 1.5}
                    }
                  },
                  font: {
                    color: '#fff'
                  }
                }}
              />
            )}
          </div>
        )}
      </div>
      <h2>What are embeddings?</h2>
      <p>The ability to do math on language is a foundational building block of AI. For that to work, we need a way to convert language into numbers while preserving meaning.
      This is a challenge that occupied researchers for decades since the 1950s until embeddings emerged as a solution.
      In simple terms, embeddings represent words/sentences as points in a high-dimensional space (i.e. vectors) where words/sentences that have similar meanings are close to each other, and words/sentences that have different meanings are far apart.</p>
      <h2>How do embeddings work?</h2>
      <p>Over the years since the basic idea was first proposed, many increasingly sophisticated methods have been developed to generate embeddings.
      Since ~2005 and then following key advances in the early 2010s, approaches based on neural networks have become dominant.
      Glossing over a lot of important details, this involves training a neural network to learn relationships between words/sentences by analyzing massive amounts of text, until it can reliably assign a vector representation to any given word/sentence that successfully captures its meaning.</p>
      <h2>Are there issues with this approach?</h2>
      <p>Yes. One very important issue to keep in mind is that although embedding generation might seem like a purely mathematical process, it is in fact far from neutral. Because the underlying neural networks are trained to recognize patterns in vast quantities of human-generated text, they inevitably pick up various biases present in that source material.
      Careless use of the outputs of such models therefore risks inadvertently perpetuating and even reinforcing potentially harmful biases in subtle yet pervasive ways.</p>
      <h2>How does this tool work?</h2>
      <p>This tool uses one such neural network ("model") to generate such vector representations from the text inputs you provide.
      This particular model maps inputs to 364-dimensional vectors. Because humans can only comfortably visualize up to 3 dimensions, we use a technique called "dimensionality reduction" to project these high-dimension vectors in 2D or 3D space.</p>

    </>
  );
}

export default App
