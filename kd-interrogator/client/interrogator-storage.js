(global => {
    const STORAGE_KEY = 'kubin-interrogator-settings'

    const defaultSettings = {
        modelIndex: 0,
        clipModel: 'ViT-L-14/openai',
        mode: 'fast',
        blipModelType: 'blip-large',
        chunkSize: 2048,
        vlmModel: 'vikhyatk/moondream2',
        vlmPrompt: 'Output the detailed description of this image.',
        quantization: '4bit',
        prependText: '',
        appendText: '',
        batchMode: 0,
        imageExtensions: ['.jpg', '.jpeg', '.png', '.bmp'],
        skipExisting: true,
        removeLineBreaks: false,
        captionExtension: '.txt',
        outputCsv: 'captions.csv'
    }

    kubin.interrogator = {
        saveSettings: (settings) => {
            try {
                localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
                console.log('Interrogator settings saved')
            } catch (e) {
                console.warn('Failed to save interrogator settings:', e)
            }
        },

        loadSettings: () => {
            try {
                const saved = localStorage.getItem(STORAGE_KEY)
                if (saved) {
                    const settings = JSON.parse(saved)
                    console.log('Interrogator settings loaded')
                    return { ...defaultSettings, ...settings }
                }
            } catch (e) {
                console.warn('Failed to load interrogator settings:', e)
            }
            return defaultSettings
        },

        clearSettings: () => {
            try {
                localStorage.removeItem(STORAGE_KEY)
                console.log('Interrogator settings cleared')
            } catch (e) {
                console.warn('Failed to clear interrogator settings:', e)
            }
        },

        getDefaultSettings: () => defaultSettings
    }
})(window)
