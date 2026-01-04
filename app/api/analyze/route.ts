import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { spawn } from 'child_process';
import { tmpdir } from 'os';

// --- Types ---
interface Annotation {
  time: string;
  text: string;
}

interface AnalysisResult {
  action_class: string;
  confidence: string;
  annotations: Annotation[];
}

// --- Configuration ---
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg', 'image/webp'];

/**
 * POST /api/analyze
 * Handles image upload and analysis
 */
export async function POST(request: NextRequest) {
  try {
    // Parse multipart form data
    const formData = await request.formData();
    const file = formData.get('image') as File;

    // --- Validation ---
    if (!file) {
      return NextResponse.json(
        { error: 'No image file provided' },
        { status: 400 }
      );
    }

    if (!ALLOWED_TYPES.includes(file.type)) {
      return NextResponse.json(
        { error: 'Invalid file type. Only image files (JPEG, PNG, WebP) are allowed.' },
        { status: 400 }
      );
    }

    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: 'File size exceeds 10MB limit' },
        { status: 400 }
      );
    }

    // --- Process Image ---
    // Convert file to buffer for processing
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    // TODO: Replace this with actual model inference
    // This is where you would:
    // 1. Preprocess the image (resize, normalize)
    // 2. Pass image through your CNN model
    // 3. Get predictions and classification results
    
    const result = await analyzeImage(buffer, file.name);

    return NextResponse.json(result, { status: 200 });

  } catch (error) {
    console.error('Error in analyze endpoint:', error);
    return NextResponse.json(
      { error: 'Internal server error during analysis' },
      { status: 500 }
    );
  }
}

/**
 * Analyzes image using the Python model
 */
async function analyzeImage(
  imageBuffer: Buffer,
  filename: string
): Promise<AnalysisResult> {
  // Create temporary file path
  const tempPath = join(tmpdir(), `upload_${Date.now()}_${filename}`);
  
  try {
    // Save buffer to temporary file
    await writeFile(tempPath, imageBuffer);
    
    // Path to Python script
    const scriptPath = join(process.cwd(), 'python_model', 'predict.py');
    
    // Call Python model
    const result = await runPythonModel(scriptPath, tempPath);
    
    // Clean up temporary file
    await unlink(tempPath);
    
    return result;
    
  } catch (error) {
    // Clean up on error
    try {
      await unlink(tempPath);
    } catch {}
    throw error;
  }
}

/**
 * Execute Python model script and parse results
 */
function runPythonModel(scriptPath: string, imagePath: string): Promise<AnalysisResult> {
  return new Promise((resolve, reject) => {
    // Use virtual environment Python
    const pythonPath = join(process.cwd(), '.venv', 'bin', 'python');
    
    // Spawn Python process
    const pythonProcess = spawn(pythonPath, [scriptPath, imagePath]);
    
    let stdout = '';
    let stderr = '';
    
    // Collect stdout
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    // Collect stderr
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    // Handle process completion
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python error:', stderr);
        reject(new Error(`Python script failed: ${stderr || 'Unknown error'}`));
        return;
      }
      
      try {
        // Parse JSON output
        const result = JSON.parse(stdout);
        
        // Check for error in result
        if (result.error) {
          reject(new Error(result.error));
          return;
        }
        
        resolve(result);
      } catch (error) {
        reject(new Error(`Failed to parse model output: ${error}`));
      }
    });
    
    // Handle process errors
    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to start Python process: ${error.message}`));
    });
  });
}

/**
 * GET /api/analyze
 * Health check endpoint
 */
export async function GET() {
  return NextResponse.json({
    status: 'ok',
    message: 'Image analysis API is running',
    version: '1.0.0'
  });
}
