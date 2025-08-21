"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { signIn, useSession } from "next-auth/react"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import Loading from "../common/Loading"

export function LoginPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [isloading, setIsLoading] = useState(false);

  useEffect(() => {
    if(session) {
        router.push("/application");
    }
  }, [session, router])

  if(status === "loading"){
    return <Loading />
  }

  if (session){
    return null
  }

  const handleGoogleLogin = async () => {
    try {
        await signIn("google", { callbackUrl: "/application" })
    } catch (error) {
        console.error("Sign in error: ", error);
    } finally{
        setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex w-screen">
      {/* Left Side - Login Form */}
      <div className="flex-1 flex items-center justify-center p-8 bg-background">
        <div className="w-full max-w-sm space-y-8 p-8">
          {/* Logo/Brand */}
          <Card className="border-border">
          <div className="text-center space-y-2">
            <div className="w-12 h-12 bg-primary rounded-lg mx-auto flex items-center justify-center">
              <svg className="w-6 h-6 text-primary-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-foreground">Welcome Back</h1>
            <p className="text-muted-foreground">Sign in to your account</p>
          </div>

          {/* Google Login Button */}
          {/* <Card className="border-border"> */}
            <CardContent className="p-6">
              <Button
                onClick={handleGoogleLogin}
                variant="outline"
                size="lg"
                className="w-full h-12 text-base font-medium border-2 hover:bg-accent hover:text-accent-foreground hover:border-accent transition-all duration-200 bg-transparent"
              >
                <svg className="w-5 h-5 mr-3" viewBox="0 0 24 24">
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </svg>
                Continue with Google
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Right Side - Animation */}
      <div className="flex-1 bg-gradient-to-br from-primary/10 via-accent/5 to-primary/20 relative overflow-hidden hidden lg:flex items-center justify-center">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-primary rounded-full animate-pulse-glow"></div>
          <div
            className="absolute top-3/4 right-1/4 w-24 h-24 bg-accent rounded-full animate-pulse-glow"
            style={{ animationDelay: "2s" }}
          ></div>
          <div
            className="absolute top-1/2 left-1/2 w-16 h-16 bg-primary rounded-full animate-pulse-glow"
            style={{ animationDelay: "4s" }}
          ></div>
        </div>

        {/* Main Animation Elements */}
        <div className="relative z-10 flex flex-col items-center space-y-8">
          {/* Floating Geometric Shapes */}
          <div className="relative">
            <div className="w-32 h-32 border-4 border-primary/30 rounded-2xl animate-float"></div>
            <div
              className="absolute -top-4 -right-4 w-16 h-16 bg-accent/20 rounded-full animate-float"
              style={{ animationDelay: "1s" }}
            ></div>
            <div
              className="absolute -bottom-4 -left-4 w-12 h-12 bg-primary/20 rounded-lg animate-float"
              style={{ animationDelay: "3s" }}
            ></div>
          </div>

          {/* Gradient Orb */}
          <div className="w-48 h-48 rounded-full bg-gradient-to-r from-primary/20 via-accent/30 to-primary/20 animate-gradient blur-sm"></div>

          {/* Text Content */}
          <div className="text-center space-y-4 max-w-md">
            <h2 className="text-3xl font-bold text-foreground">Secure & Modern</h2>
            <p className="text-lg text-muted-foreground leading-relaxed">
              Experience seamless authentication with enterprise-grade security and beautiful design.
            </p>
          </div>

          {/* Floating Icons */}
          <div className="flex space-x-8 mt-8">
            <div
              className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center animate-float"
              style={{ animationDelay: "0.5s" }}
            >
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
            </div>
            <div
              className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center animate-float"
              style={{ animationDelay: "1.5s" }}
            >
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div
              className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center animate-float"
              style={{ animationDelay: "2.5s" }}
            >
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
            </div>
          </div>
        </div>

        {/* Responsive Mobile View */}
        <div className="lg:hidden absolute inset-0 bg-gradient-to-b from-primary/5 to-accent/10"></div>
      </div>
    </div>
  )
}
