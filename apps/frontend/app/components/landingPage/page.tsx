import { Navbar } from "./Navbar";
import Hero from "./Hero";
import { FeaturesSection } from "./FeatureSection";
import { AboutSection } from "./AboutSection";

export default function LandingPage() {
  return (
    <>
      <Navbar />
      <main className="w-screen">
        <Hero />
        <FeaturesSection />
        <AboutSection />
      </main>
    </>
  );
}
