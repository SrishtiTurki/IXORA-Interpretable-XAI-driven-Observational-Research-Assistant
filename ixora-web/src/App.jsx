import { Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './context/AuthContext'
import { ChatProvider } from './context/ChatContext'
import Cursor from './components/Cursor'
import Toast from './components/Toast'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import AuthCallback from './pages/AuthCallback'

function ProtectedRoute({ children }) {
  const { isAuthed } = useAuth()
  return isAuthed ? children : <Navigate to="/login" replace />
}

function AppRoutes() {
  return (
    <>
      <Cursor />
      <Toast />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/auth/callback" element={<AuthCallback />} />
        <Route path="/*" element={
          <ProtectedRoute>
            <ChatProvider>
              <DashboardPage />
            </ChatProvider>
          </ProtectedRoute>
        } />
      </Routes>
    </>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
    </AuthProvider>
  )
}
