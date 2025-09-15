import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Lightbulb, Rocket, PieChart } from "lucide-react";

const FEATURES = [
  {
    icon: <Lightbulb className="h-8 w-8 text-primary" />,
    title: "Actionable Insights",
    desc: "Get AI-generated recommendations to improve conversions in minutes.",
  },
  {
    icon: <Rocket className="h-8 w-8 text-primary" />,
    title: "Real-Time Monitoring",
    desc: "Watch user journeys unfold live and spot friction instantly.",
  },
  {
    icon: <PieChart className="h-8 w-8 text-primary" />,
    title: "Unified Dashboard",
    desc: "All of your metrics, forecasts, and cohorts in one beautiful view.",
  },
];

export function FeaturesSection() {
  return (
    <section id="features" className="py-20">
      <div className="container px-4 sm:px-6 max-w-7xl mx-auto text-center space-y-4">
        <Badge variant="secondary" className="mx-auto">
          Why StatsAI
        </Badge>

        <h2 className="text-3xl md:text-4xl font-extrabold tracking-tight">
          Features built for scale
        </h2>

        <p className="max-w-2xl mx-auto text-muted-foreground">
          Everything you need to understand visitors, boost revenue, and grow
          confidently.
        </p>

        <div className="grid gap-6 mt-12 sm:grid-cols-2 md:grid-cols-3">
          {FEATURES.map((f) => (
            <Card key={f.title} className="text-left">
              <CardHeader className="flex items-center gap-4">
                {f.icon}
                <CardTitle>{f.title}</CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">
                {f.desc}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}