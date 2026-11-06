import { COLORS, createColorExample } from './colorData.js';

const TRAINING_EXAMPLES_PER_COLOR = 40;
const TRAINING_OPTIONS = { epochs: 40, batchSize: 12 };

// Build labeled RGB examples: TRAINING_EXAMPLES_PER_COLOR for each color,
// as objects shaped { red, green, blue, label }.
function createTrainingExamples() {
    return COLORS.flatMap((color) =>
        Array.from({ length: TRAINING_EXAMPLES_PER_COLOR }, () =>
            createColorExample(color),
        ),
    );
}
console.log(createTrainingExamples());
export default class ML {
    constructor() {
        this.model = null;
    }
    // Create labeled RGB examples, train the ml5 model, and report progress.
    async trainModel(updateProgress) {
        if (typeof ml5 === 'undefined') {
            throw new Error(
                'ml5.js is not loaded. Check your internet connection and reload the page.',
            );
        }

        const report = (message) => updateProgress?.(message);
        const totalEpochs = TRAINING_OPTIONS.epochs;

        // Clear any old model so a failed retrain cannot leave a stale one.
        this.model = null;

        report('Creating training data...');
        const model = ml5.neuralNetwork({ task: 'classification' });
        for (const { red, green, blue, label } of createTrainingExamples()) {
            model.addData([red, green, blue], [label]);
        }
        model.normalizeData();

        report('Training model...');
        await new Promise((resolve, reject) => {
            const whileTraining = (epoch) => {
                const done = Math.min(epoch + 1, totalEpochs);
                report(`Training model... epoch ${done} of ${totalEpochs}`);
            };
            // Resolve only when ml5 says training is finished. If train()
            // throws, or returns a promise that rejects, the error is passed
            // on to the caller.
            const pending = model.train(
                { ...TRAINING_OPTIONS },
                whileTraining,
                resolve,
            );
            pending?.catch?.(reject);
        });

        this.model = model;
    }

    // Classify an RGB color with the trained model and return its label.

    async classify(color) {
        if (!this.model) {
            throw new Error(
                'The model has not been trained yet. Train the model first.',
            );
        }

        // Use only the RGB values. The game's color objects also carry the
        // true label, which the model must never see.
        const { red, green, blue } = color ?? {};
        const input = [red, green, blue];
        if (!input.every(Number.isFinite)) {
            throw new Error(
                'classify() needs numeric red, green, and blue values.',
            );
        }

        const results = await new Promise((resolve, reject) => {
            // ml5 v1 calls back with (results, error).
            const pending = this.model.classify(input, (results, error) => {
                if (error) {
                    reject(error);
                } else {
                    resolve(results);
                }
            });
            pending?.catch?.(reject);
        });

        // Results are sorted by confidence, so the first entry is the best guess.
        const label = results?.[0]?.label;
        if (typeof label !== 'string') {
            throw new Error('The model did not return a prediction.');
        }
        return label.toLowerCase();
    }


}
