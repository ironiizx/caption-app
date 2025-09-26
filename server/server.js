import 'dotenv/config';
import express from 'express';
import { pipeline } from '@xenova/transformers';
import sharp from 'sharp';
import fetch from 'node-fetch'; // Es posible que necesites instalarlo: npm install node-fetch

const PORT = process.env.PORT || 3000;
const MODEL_ID = process.env.MODEL_ID || 'Xenova/vit-gpt2-image-captioning';

const app = express();
app.use(express.json({ limit: '20mb' }));

// CORS simple
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

app.post('/caption', async (req, res) => {
    try {
        const { image_url, max_new_tokens = 30 } = req.body || {};
        if (!image_url) {
            return res.status(400).json({ error: 'Falta image_url' });
        }
        
        const pipe = await getPipe();
        const input = String(image_url);

        // Descargamos la imagen para las métricas
        const imageResponse = await fetch(input);
        if (!imageResponse.ok) throw new Error(`Failed to fetch image: ${imageResponse.statusText}`);
        const imageBuffer = Buffer.from(await imageResponse.arrayBuffer());

        const [captionResult, metricsResult] = await Promise.all([
            pipe(input, { max_new_tokens }),
            (async () => {
                const image = sharp(imageBuffer);
                const metadata = await image.metadata();
                const stats = await image.stats();
                const brightness = (stats.channels[0].mean + stats.channels[1].mean + stats.channels[2].mean) / 3;
                return {
                    width: metadata.width,
                    height: metadata.height,
                    format: metadata.format,
                    brightness: parseFloat(brightness.toFixed(2)),
                };
            })()
        ]);
        
        const caption = captionResult?.[0]?.generated_text ?? '';
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