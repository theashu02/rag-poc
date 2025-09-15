import Image from "next/image";
import { Button } from "@/components/ui/button";

export function AboutSection() {
  return (
    <section id="about" className="py-20">
      <div className="container px-4 sm:px-6 max-w-7xl mx-auto grid gap-12 md:grid-cols-2 items-center">
        {/* Illustration */}
        <div className="relative h-48 sm:h-64 md:h-full">
          <Image
            src="/illustrations/analytics-dashboard.png"
            alt="Dashboard"
            fill
            className="object-contain"
            sizes="(max-width: 768px) 100vw, 50vw"
          />
        </div>

        {/* Copy */}
        <div className="space-y-6">
          <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight">
            Built by growth engineers, for growth engineers
          </h2>
          <p className="text-muted-foreground">
            StatsAI began as an internal tool to help SaaS teams unlock hidden
            revenue opportunities. Today, hundreds of companies rely on our AI
            to optimise onboarding flows, pricing pages, and everything in
            between.
          </p>
          <Button size="lg">Read our story</Button>
        </div>
      </div>
    </section>
  );
}