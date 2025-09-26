import 'dotenv/config';
import express from 'express';
import { pipeline } from '@xenova/transformers';
import sharp from 'sharp';

const PORT = process.env.PORT || 3000;
const MODEL_ID = process.env.MODEL_ID || 'Salesforce/blip-image-captioning-large';

const app = express();
app.use(express.json({ limit: '20mb' }));

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
        const { image_url, image_base64, max_new_tokens = 30 } = req.body || {};
        const pipe = await getPipe();

        let input;
        let imageBuffer; 

        if (image_url) {
            input = String(image_url);
            const imageResponse = await fetch(image_url);
            imageBuffer = Buffer.from(await imageResponse.arrayBuffer());
        } else if (image_base64) {
            imageBuffer = Buffer.from(image_base64, 'base64');
            input = new Uint8Array(imageBuffer);
        } else {
            return res.status(400).json({ error: 'Falta image_url o image_base64' });
        }

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
                    channels: stats.channels.length,
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