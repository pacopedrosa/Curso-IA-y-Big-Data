import { useState, useEffect } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [labels, setLabels] = useState([])
  const [confianzas, setConfianzas] = useState({})
  const [s3Url, setS3Url] = useState(null)
  const [historial, setHistorial] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [estadisticas, setEstadisticas] = useState(null)

  const cargarDatos = () => {
    fetch('http://localhost:8000/imagenes')
      .then(r => r.json())
      .then(setHistorial)
      .catch(() => {})
    fetch('http://localhost:8000/estadisticas')
      .then(r => r.json())
      .then(setEstadisticas)
      .catch(() => {})
  }

  useEffect(() => {
    cargarDatos()
  }, [])

  const handleFile = (e) => {
    const f = e.target.files[0]
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setLabels([])
    setS3Url(null)
    setError(null)
  }

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      setLabels(data.labels)
      setConfianzas(data.confianzas || {})
      setS3Url(data.s3_url)

      const hist = await fetch('http://localhost:8000/imagenes').then(r => r.json())
      setHistorial(hist)
      const stats = await fetch('http://localhost:8000/estadisticas').then(r => r.json())
      setEstadisticas(stats)
    } catch (e) {
      setError('Error al conectar con el servidor.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1 style={styles.title}>🏷️ FastRetail — Clasificador de Imágenes</h1>
        <p style={styles.subtitle}>Sistema automático con Amazon Rekognition + AWS S3</p>
      </header>

      <main style={styles.main}>
        <div style={styles.card}>
          <h2 style={styles.cardTitle}>📤 Subir imagen</h2>

          <label style={styles.uploadArea}>
            {preview
              ? <img src={preview} alt="preview" style={styles.preview} />
              : <span style={styles.uploadPlaceholder}>📂 Haz clic para seleccionar una imagen</span>
            }
            <input type="file" accept="image/*" onChange={handleFile} style={{ display: 'none' }} />
          </label>

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            style={!file || loading ? styles.btnDisabled : styles.btn}
          >
            {loading ? '⏳ Analizando...' : '🚀 Analizar con Rekognition'}
          </button>

          {error && <p style={styles.error}>{error}</p>}

          {labels.length > 0 && (
            <div style={styles.results}>
              <h3>✅ Etiquetas detectadas:</h3>
              <div style={styles.tags}>
                {labels.map((l, i) => (
                  <span key={i} style={styles.tag}>
                    {l} {confianzas[l] ? <small style={{opacity:0.7}}>({confianzas[l]}%)</small> : null}
                  </span>
                ))}
              </div>
              {s3Url && (
                <p style={styles.s3}>
                  ☁️ Guardado en S3: <a href={s3Url} target="_blank" rel="noreferrer" style={styles.link}>ver imagen</a>
                </p>
              )}
            </div>
          )}
        </div>

        <div style={styles.card}>
          <h2 style={styles.cardTitle}>📋 Historial de clasificaciones</h2>
          {historial.length === 0
            ? <p style={{ color: '#888' }}>Aún no hay imágenes clasificadas.</p>
            : historial.map(img => (
              <div key={img.id} style={styles.histItem}>
                <strong>{img.nombre}</strong>
                <div style={styles.tags}>
                  {img.etiquetas.split(', ').map((e, i) => (
                    <span key={i} style={styles.tagSmall}>{e}</span>
                  ))}
                </div>
                {img.s3_url && (
                  <a href={img.s3_url} target="_blank" rel="noreferrer" style={styles.link}>
                    ☁️ Ver en S3
                  </a>
                )}
              </div>
            ))
          }
        </div>

        {estadisticas && (
          <div style={{...styles.card, width: '100%', flexBasis: '100%'}}>
            <h2 style={styles.cardTitle}>📊 Evaluación del sistema</h2>
            <div style={styles.statsGrid}>
              <div style={styles.statBox}>
                <span style={styles.statNum}>{estadisticas.total_imagenes_procesadas}</span>
                <span style={styles.statLabel}>Imágenes procesadas</span>
              </div>
              <div style={styles.statBox}>
                <span style={styles.statNum}>{estadisticas.umbral_confianza_minima}%</span>
                <span style={styles.statLabel}>Confianza mínima</span>
              </div>
              <div style={styles.statBox}>
                <span style={styles.statNum}>{estadisticas.max_etiquetas_por_imagen}</span>
                <span style={styles.statLabel}>Máx. etiquetas/imagen</span>
              </div>
            </div>
            {estadisticas.etiquetas_mas_frecuentes.length > 0 && (
              <>
                <h3 style={{marginTop: '1rem'}}>🏆 Etiquetas más frecuentes</h3>
                <div style={styles.barChart}>
                  {estadisticas.etiquetas_mas_frecuentes.map((e, i) => (
                    <div key={i} style={styles.barRow}>
                      <span style={styles.barLabel}>{e.etiqueta}</span>
                      <div style={styles.barBg}>
                        <div style={{
                          ...styles.barFill,
                          width: `${(e.frecuencia / estadisticas.etiquetas_mas_frecuentes[0].frecuencia) * 100}%`
                        }} />
                      </div>
                      <span style={styles.barCount}>{e.frecuencia}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}
      </main>
    </div>
  )
}

const styles = {
  app: { fontFamily: 'sans-serif', minHeight: '100vh', background: '#f5f6fa' },
  header: { background: '#1a1a2e', color: 'white', padding: '2rem', textAlign: 'center' },
  title: { margin: 0, fontSize: '1.8rem' },
  subtitle: { margin: '0.5rem 0 0', color: '#aaa' },
  main: { display: 'flex', gap: '2rem', padding: '2rem', maxWidth: '1000px', margin: '0 auto', flexWrap: 'wrap' },
  card: { background: 'white', borderRadius: '12px', padding: '1.5rem', flex: 1, minWidth: '300px', boxShadow: '0 2px 10px rgba(0,0,0,0.08)' },
  cardTitle: { marginTop: 0, color: '#1a1a2e' },
  uploadArea: { display: 'flex', border: '2px dashed #ccc', borderRadius: '8px', padding: '1rem', cursor: 'pointer', textAlign: 'center', marginBottom: '1rem', minHeight: '150px', alignItems: 'center', justifyContent: 'center' },
  uploadPlaceholder: { color: '#888' },
  preview: { maxWidth: '100%', maxHeight: '200px', borderRadius: '6px' },
  btn: { width: '100%', padding: '0.75rem', background: '#e94560', color: 'white', border: 'none', borderRadius: '8px', fontSize: '1rem', cursor: 'pointer' },
  btnDisabled: { width: '100%', padding: '0.75rem', background: '#ccc', color: 'white', border: 'none', borderRadius: '8px', fontSize: '1rem', cursor: 'not-allowed' },
  results: { marginTop: '1rem' },
  tags: { display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' },
  tag: { background: '#e94560', color: 'white', padding: '0.3rem 0.8rem', borderRadius: '20px', fontSize: '0.9rem' },
  tagSmall: { background: '#eef', color: '#333', padding: '0.2rem 0.6rem', borderRadius: '20px', fontSize: '0.8rem' },
  error: { color: 'red', marginTop: '0.5rem' },
  histItem: { borderBottom: '1px solid #eee', paddingBottom: '0.75rem', marginBottom: '0.75rem' },
  s3: { fontSize: '0.85rem', color: '#555', marginTop: '0.5rem' },
  link: { color: '#e94560' },
  statsGrid: { display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '0.5rem' },
  statBox: { background: '#f0f4ff', borderRadius: '10px', padding: '1rem 1.5rem', textAlign: 'center', flex: 1, minWidth: '120px', display: 'flex', flexDirection: 'column', gap: '0.3rem' },
  statNum: { fontSize: '2rem', fontWeight: 'bold', color: '#1a1a2e' },
  statLabel: { fontSize: '0.8rem', color: '#666' },
  barChart: { display: 'flex', flexDirection: 'column', gap: '0.4rem' },
  barRow: { display: 'flex', alignItems: 'center', gap: '0.5rem' },
  barLabel: { width: '130px', fontSize: '0.85rem', textAlign: 'right', color: '#444' },
  barBg: { flex: 1, background: '#eee', borderRadius: '4px', height: '18px', overflow: 'hidden' },
  barFill: { height: '100%', background: '#e94560', borderRadius: '4px', transition: 'width 0.4s' },
  barCount: { width: '24px', fontSize: '0.8rem', color: '#888' },
}

export default App