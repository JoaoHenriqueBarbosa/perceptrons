import { createRouter, createRoute, createRootRoute, Outlet } from '@tanstack/react-router'
import { Home } from '../pages/Home'
import { Dataset } from '../pages/Dataset'
import { Training } from '../pages/Training'
import { Test } from '../pages/Test'

const rootRoute = createRootRoute({
  component: () => (
    <div className="app">
      <h1>Perceptrons</h1>
      <Outlet />
    </div>
  ),
})

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: Home,
})

const datasetRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/dataset',
  component: Dataset,
})

const trainingRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/training',
  component: Training,
})

const testRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/test',
  component: Test,
})

const routeTree = rootRoute.addChildren([indexRoute, datasetRoute, trainingRoute, testRoute])

export const router = createRouter({ routeTree })

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router
  }
}
