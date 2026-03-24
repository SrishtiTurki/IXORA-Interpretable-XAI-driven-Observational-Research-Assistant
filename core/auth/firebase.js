// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBXKNAucnsRagbZQgzJeVY3df-JlVaTwSE",
  authDomain: "ixora-6b2a0.firebaseapp.com",
  projectId: "ixora-6b2a0",
  storageBucket: "ixora-6b2a0.firebasestorage.app",
  messagingSenderId: "415982001170",
  appId: "1:415982001170:web:f9cc9bca68a21d04528ccb",
  measurementId: "G-B4PV7K13M1"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);