const workerCode = `
  self.onmessage = ({ data }) => {
    if (typeof data !== 'string') {
      self.postMessage({ error: 'Input must be a string' });
      return;
    }

    let s = data.normalize('NFD');
    s = s.replace(/[\\u0300-\\u036f]/g, '');
    s = s.replace(/[^\\p{L}\\p{N}\\s]/gu, '');
    s = s.replace(/\\s+/g, ' ').trim();
    s = s.toLowerCase();

    self.postMessage({ result: s });
  };
`;

const blob = new Blob([workerCode], { type: "application/javascript" });
const worker = new Worker(URL.createObjectURL(blob));

export function clean(text: string): Promise<string> {
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) =>
      data.error ? reject(new TypeError(data.error)) : resolve(data.result);
    worker.onerror = reject;
    worker.postMessage(text);
  });
}
