import 'dotenv/config';
import express from 'express';
import { pipeline } from '@xenova/transformers';
import sharp from 'sharp'; // NUEVO: Importamos sharp para análisis de imagen

const PORT = process.env.PORT || 3000;
const MODEL_ID = process.env.MODEL_ID || 'Salesforce/blip-image-captioning-large';

const app = express();
app.use(express.json({ limit: '20mb' }));

// CORS simple (sin cambios)
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', req.headers.origin || '*');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
});

let pipePromise;
async function getPipe() {
    if (!pipePromise) {
        pipePromise = pipeline('image-to-text', MODEL_ID);
    }
    return pipePromise;
}

// POST /caption -> { image_url } o { image_base64 }
app.post('/caption', async (req, res) => {
    try {
        const { image_url, image_base64, max_new_tokens = 30 } = req.body || {};
        const pipe = await getPipe();

        let input;
        let imageBuffer; // NUEVO: Buffer para analizar con sharp

        if (image_url) {
            input = String(image_url);
            // NUEVO: Descargamos la imagen para analizarla
            const imageResponse = await fetch(image_url);
            imageBuffer = Buffer.from(await imageResponse.arrayBuffer());
        } else if (image_base64) {
            imageBuffer = Buffer.from(image_base64, 'base64');
            input = new Uint8Array(imageBuffer);
        } else {
            return res.status(400).json({ error: 'Falta image_url o image_base64' });
        }

        // --- Ejecutamos ambas tareas en paralelo para más velocidad ---
        const [captionResult, metricsResult] = await Promise.all([
            // Tarea 1: Generar caption (sin cambios)
            pipe(input, { max_new_tokens }),
            
            // TAREA 2: Calcular métricas con sharp
            (async () => {
                const image = sharp(imageBuffer);
                const metadata = await image.metadata();
                const stats = await image.stats();
                
                // Calculamos un valor simple de "brillo" (promedio de los canales R,G,B)
                const brightness = (stats.channels[0].mean + stats.channels[1].mean + stats.channels[2].mean) / 3;

                return {
                    width: metadata.width,
                    height: metadata.height,
                    format: metadata.format,
                    brightness: parseFloat(brightness.toFixed(2)), // Brillo en una escala de 0 a 255
                    channels: stats.channels.length,
                };
            })()
        ]);
        
        const caption = captionResult?.[0]?.generated_text ?? '';

        // Devolvemos el caption Y las nuevas métricas
        res.json({ caption, model: MODEL_ID, metrics: metricsResult });

    } catch (err) {
        console.error(err);
        res.status(500).json({ error: String(err) });
    }
});

app.listen(PORT, () => {
    console.log(`✅ Caption server on http://localhost:${PORT}`);
    console.log(`   Modelo: ${MODEL_ID}`);
});