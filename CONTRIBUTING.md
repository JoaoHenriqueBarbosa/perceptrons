# Contribuindo para o Projeto Perceptrons

Obrigado por considerar contribuir para este projeto educacional! 🎉

## Como Contribuir

### 1. Reportar Bugs

Se você encontrou um bug:

1. Verifique se o bug já foi reportado nas [Issues](https://github.com/JoaoHenriqueBarbosa/perceptrons/issues)
2. Se não, crie uma nova issue incluindo:
   - Descrição clara do problema
   - Passos para reproduzir
   - Comportamento esperado vs. atual
   - Screenshots se aplicável
   - Informações de ambiente (OS, Python version, etc.)

### 2. Sugerir Features

Para sugerir novas funcionalidades:

1. Abra uma [Issue](https://github.com/JoaoHenriqueBarbosa/perceptrons/issues) com o título começando com `[Feature Request]`
2. Descreva:
   - O problema que a feature resolve
   - Como você imagina que funcionaria
   - Exemplos de uso
   - Por que isso seria útil para aprendizado

### 3. Enviar Pull Requests

#### Setup do Ambiente

```bash
# Fork o repositório no GitHub
# Clone seu fork
git clone https://github.com/SEU-USUARIO/perceptrons.git
cd perceptrons

# Adicione o upstream
git remote add upstream https://github.com/JoaoHenriqueBarbosa/perceptrons.git

# Crie uma branch para sua feature
git checkout -b feature/nome-da-feature
```

#### Workflow

1. **Faça suas mudanças**
   - Mantenha commits pequenos e focados
   - Escreva mensagens de commit descritivas
   - Teste suas mudanças

2. **Teste localmente**
   ```bash
   # Backend
   cd backend
   source venv/bin/activate
   python -m pytest  # se houver testes

   # Frontend
   cd frontend
   bun run build  # verifica se builda
   ```

3. **Commit e Push**
   ```bash
   git add .
   git commit -m "feat: adiciona nova funcionalidade X"
   git push origin feature/nome-da-feature
   ```

4. **Abra um Pull Request**
   - Use um título descritivo
   - Explique o que mudou e por quê
   - Referencie issues relacionadas

### Convenções de Código

#### Python (Backend)
```python
# Use type hints
def train(self, X: np.ndarray, y: np.ndarray, epochs: int) -> List[float]:
    pass

# Docstrings para funções públicas
def predict(self, X: np.ndarray) -> int:
    """
    Faz a predição para uma entrada.

    Args:
        X: Array de features (1D array com 256 elementos)

    Returns:
        Classe predita (0 ou 1)
    """
    pass
```

#### TypeScript (Frontend)
```typescript
// Use tipos explícitos
interface Pattern {
  id: number
  pattern: boolean[][]
  label: string
}

// Componentes funcionais com tipos
export function Dataset(): JSX.Element {
  // ...
}
```

#### Commits

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: adiciona visualização de pesos
fix: corrige bug na predição
docs: atualiza README com novos exemplos
style: formata código Python
refactor: reorganiza estrutura de componentes
test: adiciona testes para MultiPerceptron
chore: atualiza dependências
```

### Ideias de Contribuição

#### Features Educacionais
- [ ] Visualização interativa dos pesos aprendidos
- [ ] Animação do processo de treinamento época por época
- [ ] Explicação visual de separabilidade linear
- [ ] Matriz de confusão após treinamento
- [ ] Gráficos de erro vs. épocas

#### Melhorias Técnicas
- [ ] Implementar MLP (Multi-Layer Perceptron)
- [ ] Adicionar validação cruzada
- [ ] Train/test split automático
- [ ] Data augmentation para patterns
- [ ] Export/import de datasets

#### UX/UI
- [ ] Dark/light mode toggle manual
- [ ] Arrastar para reorganizar patterns
- [ ] Editar patterns salvos
- [ ] Presets de datasets (XOR, AND, OR)
- [ ] Tutorial interativo para novos usuários

#### Documentação
- [ ] Adicionar mais experimentos ao README
- [ ] Vídeos demonstrativos
- [ ] Explicações matemáticas mais detalhadas
- [ ] Tradução para outros idiomas

## Diretrizes

### O que NÃO aceitar

- Mudanças que complicam demais o código (lembre-se: é educacional!)
- Features que fogem do escopo de perceptrons básicos
- Dependências pesadas desnecessárias
- Código sem documentação

### O que SEMPRE é bem-vindo

- Melhorias na clareza do código
- Mais exemplos e experimentos educacionais
- Correções de bugs
- Melhorias de performance sem complicar
- Documentação mais clara

## Código de Conduta

- Seja respeitoso e construtivo
- Assuma boas intenções
- Aceite feedback com mente aberta
- Foque no aprendizado, não na perfeição

## Dúvidas?

Abra uma [Issue](https://github.com/JoaoHenriqueBarbosa/perceptrons/issues) com sua dúvida!

---

**Obrigado por contribuir! Juntos tornamos o aprendizado de ML mais acessível! 🚀**
