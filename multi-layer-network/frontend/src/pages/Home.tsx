import { Link } from '@tanstack/react-router'

export function Home() {
  return (
    <div className="page home-page">
      <h2>Bem-vindo ao Perceptrons</h2>
      <p>Sistema de treinamento de redes neurais simples</p>
      <nav className="home-nav">
        <Link to="/dataset" className="nav-card">
          <h3>Dataset</h3>
          <p>Criar e gerenciar exemplos de treinamento</p>
        </Link>
        <Link to="/training" className="nav-card">
          <h3>Training</h3>
          <p>Treinar modelos com múltiplos neurônios</p>
        </Link>
        <Link to="/test" className="nav-card">
          <h3>Test</h3>
          <p>Testar padrões com modelos treinados</p>
        </Link>
      </nav>
    </div>
  )
}
