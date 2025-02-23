import { pipeline, env, PipelineType } from '@xenova/transformers';

env.allowLocalModels = false;

class MyEmbeddingPipeline {
  static task: PipelineType = 'feature-extraction';
  static model = 'Xenova/all-MiniLM-L6-v2';
  static instance = null;

  static async getInstance(progress_callback = null) {
    console.log('getInstance called, instance status:', this.instance);
    if (this.instance === null) {
      console.log('Creating new pipeline instance...');
      this.instance = await pipeline(this.task, this.model, { progress_callback });
      console.log('Pipeline instance created successfully');
    }
    return this.instance;
  }
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
  try {
    console.log('Worker received message:', event.data);
    
    if (event.data.type === 'load') {
      console.log('Starting model loading...');
      await MyEmbeddingPipeline.getInstance((data) => {
        console.log('Loading progress:', data);
        // Transform the progress data into a more useful format
        if (data.status === 'progress') {
          self.postMessage({
            status: 'loading',
            file: data.name,
            progress: (data.loaded / data.total) * 100
          });
        } else if (data.status === 'download') {
          self.postMessage({
            status: 'loading',
            file: data.name,
            progress: 0
          });
        }
      });
      console.log('Model loading complete');
      self.postMessage({ status: 'ready' });
      return;
    }

    // Get the pipeline instance for embedding generation
    const embedder = await MyEmbeddingPipeline.getInstance();
    const rawOutput = await embedder(event.data.texts, { pooling: 'mean', normalize: true });
    
    // Convert tensor output to regular arrays using tolist()
    const output = rawOutput.tolist();

    self.postMessage({
      status: 'complete',
      output,
      texts: event.data.texts
    });
  } catch (error) {
    console.error('Worker error:', error);
    self.postMessage({
      status: 'error',
      error: error.message
    });
  }
}); 