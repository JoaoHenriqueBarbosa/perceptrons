import { Link } from '@tanstack/react-router'

export function Home() {
  return (
    <div className="page home-page">
      <h2>Bem-vindo ao MLP com Human-in-the-Loop</h2>
      <p>Sistema de treinamento de redes neurais com feedback humano (HITL)</p>
      <nav className="home-nav">
        <Link to="/dataset" className="nav-card">
          <h3>📊 Dataset</h3>
          <p>Criar e gerenciar exemplos de treinamento</p>
        </Link>
        <Link to="/training" className="nav-card">
          <h3>🎓 Training</h3>
          <p>Treinar modelos MLP com backpropagation</p>
        </Link>
        <Link to="/test" className="nav-card">
          <h3>🧪 Test</h3>
          <p>Testar padrões com modelos treinados</p>
        </Link>
        <Link to="/test-hitl" className="nav-card hitl-card">
          <h3>🎯 Test HITL</h3>
          <p>Testar com feedback humano (Active Learning + RLHF)</p>
          <span className="badge">NOVO</span>
        </Link>
      </nav>

      <div className="hitl-features">
        <h3>✨ Recursos Human-in-the-Loop</h3>
        <ul>
          <li><strong>Active Learning:</strong> Sistema pede feedback quando incerto</li>
          <li><strong>RLHF:</strong> Ajuste de pesos baseado em correções humanas</li>
          <li><strong>Aprendizado Contínuo:</strong> Modelo evolui com feedback</li>
        </ul>
      </div>
    </div>
  )
}
