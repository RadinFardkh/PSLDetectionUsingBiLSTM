import { useEffect, useRef, useState } from 'react'
import { CameraView, useCameraPermissions } from 'expo-camera'
import { StatusBar } from 'expo-status-bar'
import { ActivityIndicator, Pressable, SafeAreaView, StyleSheet, Text, View } from 'react-native'
import { loadLocalModel, type LocalInferenceAdapter, type ModelStatus } from './src/model'

export default function App() {
  const [permission, requestPermission] = useCameraPermissions()
  const [running, setRunning] = useState(false)
  const [developerMode, setDeveloperMode] = useState(false)
  const [model, setModel] = useState<ModelStatus | null>(null)
  const inferenceRef = useRef<LocalInferenceAdapter | null>(null)
  const [message, setMessage] = useState('برای شروع، دوربین را فعال کنید')
  const [sentence, setSentence] = useState<string[]>([])
  const cameraRef = useRef<CameraView>(null)

  useEffect(() => {
    loadLocalModel().then((loaded) => {
      inferenceRef.current = loaded
      setModel(loaded.status)
    }).catch((error: Error) => setMessage(`خطا در مدل: ${error.message}`))
  }, [])
  const toggleCamera = async () => {
    if (!permission?.granted) { await requestPermission(); return }
    setRunning((value) => !value)
    setMessage(running ? 'آماده' : 'تشخیص فعال')
  }

  return <SafeAreaView style={styles.screen}><StatusBar style="light" />
    <View style={styles.header}><View><Text style={styles.eyebrow}>دستیار هوشمند</Text><Text style={styles.title}>مترجم زبان اشاره</Text></View><Pressable accessibilityRole="button" onPress={() => setDeveloperMode((value) => !value)} style={[styles.devButton, developerMode && styles.devButtonActive]}><Text style={styles.devText}>حالت توسعه</Text></Pressable></View>
    <View style={styles.cameraCard}>{running ? <CameraView ref={cameraRef} facing="front" style={styles.camera} /> : <View style={styles.placeholder}><Text style={styles.cameraIcon}>⌾</Text><Text style={styles.placeholderText}>{model ? 'برای شروع تشخیص، دکمه پایین را بزنید' : 'در حال آماده‌سازی مدل...'}</Text>{!model && <ActivityIndicator color="#54a9ff" />}</View>}{developerMode && <View pointerEvents="none" style={styles.overlay}><Text style={styles.overlayText}>LANDMARKS • 180 FEATURES</Text></View>}</View>
    <View style={styles.prediction}><Text style={styles.predictionLabel}>حرکت فعلی</Text><Text style={styles.predictionWord}>{running ? 'در حال تحلیل…' : '—'}</Text>{developerMode && <Text style={styles.confidence}>اعتماد: —</Text>}</View>
    <View style={styles.sentenceCard}><View style={styles.sentenceHeader}><Text style={styles.cardTitle}>جمله شما</Text><Pressable onPress={() => setSentence([])}><Text style={styles.clear}>پاک کردن</Text></Pressable></View><Text style={styles.sentenceText}>{sentence.length ? sentence.join(' ') : 'کلمات شناسایی‌شده اینجا نمایش داده می‌شوند'}</Text></View>
    <View style={styles.footer}><View><Text style={styles.status}>{message}</Text><Text style={styles.substatus}>{model?.ready ? 'مدل روی دستگاه آماده است' : 'در انتظار مدل محلی'}</Text></View><Pressable onPress={toggleCamera} style={[styles.startButton, running && styles.stopButton]}><Text style={styles.startText}>{running ? 'توقف' : 'شروع تشخیص'}</Text></Pressable></View>
  </SafeAreaView>
}
const styles = StyleSheet.create({ screen:{flex:1,backgroundColor:'#07111f',padding:20},header:{flexDirection:'row',justifyContent:'space-between',alignItems:'flex-start',marginBottom:18},eyebrow:{color:'#7890aa',fontSize:12,textAlign:'right'},title:{color:'#f4f8fc',fontSize:24,fontWeight:'800',textAlign:'right',marginTop:3},devButton:{backgroundColor:'#172438',paddingHorizontal:14,paddingVertical:10,borderRadius:14},devButtonActive:{backgroundColor:'#7046b8'},devText:{color:'#dfe9f5',fontWeight:'700'},cameraCard:{height:300,borderRadius:24,overflow:'hidden',backgroundColor:'#0e1b2b',borderWidth:1,borderColor:'#1b3048'},camera:{flex:1,transform:[{scaleX:-1}]},placeholder:{flex:1,justifyContent:'center',alignItems:'center',gap:12},cameraIcon:{color:'#4e6c89',fontSize:54},placeholderText:{color:'#91a5ba',textAlign:'center'},overlay:{position:'absolute',top:14,left:14,right:14,bottom:14,borderWidth:1,borderColor:'#b378ff',borderRadius:18,justifyContent:'flex-start',padding:12},overlayText:{color:'#d3aaff',fontSize:10,letterSpacing:1},prediction:{alignItems:'center',paddingVertical:18},predictionLabel:{color:'#7890aa',fontSize:13},predictionWord:{color:'#f6fbff',fontSize:32,fontWeight:'800',marginTop:4},confidence:{color:'#b378ff',fontSize:12},sentenceCard:{backgroundColor:'#0e1b2b',borderRadius:20,padding:18,minHeight:115,borderWidth:1,borderColor:'#182c42'},sentenceHeader:{flexDirection:'row',justifyContent:'space-between',alignItems:'center'},cardTitle:{color:'#eff6fc',fontSize:16,fontWeight:'800'},clear:{color:'#54a9ff',fontWeight:'700'},sentenceText:{color:'#8197ad',fontSize:15,textAlign:'right',marginTop:18,lineHeight:25},footer:{marginTop:'auto',paddingTop:18,flexDirection:'row',justifyContent:'space-between',alignItems:'center'},status:{color:'#d7e5f2',fontWeight:'700',textAlign:'right'},substatus:{color:'#6f879f',fontSize:11,textAlign:'right',marginTop:4},startButton:{backgroundColor:'#2387e8',paddingHorizontal:22,paddingVertical:15,borderRadius:16},stopButton:{backgroundColor:'#c84b5d'},startText:{color:'#fff',fontWeight:'800',fontSize:15} })
